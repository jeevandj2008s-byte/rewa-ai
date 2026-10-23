"""detector.py - Multi-Object Detection for ReWA.AI

Two-stage computer vision pipeline:
1. Stage 1: Fast, class-agnostic YOLO detection (Ultralytics) to isolate individual objects.
2. Stage 2: Target-specific CLIP classification on each object crop.

Features:
- Ultralight YOLOv8n / YOLO11n model optimized for CPU and Streamlit Community Cloud (~1 GB RAM).
- Class-agnostic detection with low confidence threshold (~0.18) so unlabelled waste items are detected.
- Aspect-ratio preserving resizing (longest side 640) for fast inference (<60ms on CPU).
- Coordinate clamping, adaptive padding, noise suppression, containment merging, and left-to-right sorting.
- High-contrast visual annotation with numbered badges.
- Complete graceful fallback to whole-image classification when YOLO finds no objects or fails to load.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# Color palette for object badges (high contrast on light and dark backgrounds)
BADGE_COLORS = [
    "#22C55E",  # Emerald Green (#1)
    "#F59E0B",  # Amber (#2)
    "#0EA5E9",  # Sky Blue (#3)
    "#EC4899",  # Pink (#4)
    "#8B5CF6",  # Violet (#5)
    "#14B8A6",  # Teal (#6)
]


# =====================================================================
# GEOMETRY & BOUNDING BOX HELPERS (PURE PYTHON & FULLY TESTABLE)
# =====================================================================
def clamp_box(box: List[int], img_w: int, img_h: int) -> List[int]:
    """Ensure box coordinates [x1, y1, x2, y2] strictly lie within image dimensions."""
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), img_w - 1))
    y1 = max(0, min(int(y1), img_h - 1))
    x2 = max(x1 + 1, min(int(x2), img_w))
    y2 = max(y1 + 1, min(int(y2), img_h))
    return [x1, y1, x2, y2]


def pad_box(box: List[int], img_w: int, img_h: int, padding_ratio: float = 0.10) -> List[int]:
    """
    Expand bounding box by padding_ratio around edges so crops are not too tight,
    then clamp inside image boundaries.
    """
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    dx = int(w * padding_ratio)
    dy = int(h * padding_ratio)

    return clamp_box([x1 - dx, y1 - dy, x2 + dx, y2 + dy], img_w, img_h)


def calculate_iou(box1: List[int], box2: List[int]) -> float:
    """Compute Intersection over Union (IoU) between two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def calculate_containment(box1: List[int], box2: List[int]) -> float:
    """
    Compute intersection over minimum area of the two boxes.
    Returns 1.0 if one box is completely contained inside the other.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    min_area = min(area1, area2)

    if min_area <= 0:
        return 0.0
    return inter_area / min_area


def merge_overlapping_boxes(
    detections: List[Dict[str, Any]],
    iou_thresh: float = 0.60,
    containment_thresh: float = 0.85,
) -> List[Dict[str, Any]]:
    """
    Merge boxes with high overlap or containment to prevent splitting a single item
    (e.g., bottle body + label) into multiple confusing boxes.
    """
    if len(detections) <= 1:
        return detections

    merged: List[Dict[str, Any]] = []
    consumed = [False] * len(detections)

    for i in range(len(detections)):
        if consumed[i]:
            continue
        curr_box = list(detections[i]["box"])
        curr_conf = float(detections[i].get("conf", 1.0))

        for j in range(i + 1, len(detections)):
            if consumed[j]:
                continue
            other_box = detections[j]["box"]
            iou = calculate_iou(curr_box, other_box)
            containment = calculate_containment(curr_box, other_box)

            if iou >= iou_thresh or containment >= containment_thresh:
                # Merge into enclosing union box
                curr_box = [
                    min(curr_box[0], other_box[0]),
                    min(curr_box[1], other_box[1]),
                    max(curr_box[2], other_box[2]),
                    max(curr_box[3], other_box[3]),
                ]
                curr_conf = max(curr_conf, float(detections[j].get("conf", 1.0)))
                consumed[j] = True

        consumed[i] = True
        merged.append({
            "box": curr_box,
            "conf": curr_conf,
        })

    return merged


def filter_and_cap_boxes(
    boxes_with_conf: List[Dict[str, Any]],
    img_w: int,
    img_h: int,
    min_size: int = 24,
    min_area_ratio: float = 0.005,
    max_objects: int = 6,
) -> List[Dict[str, Any]]:
    """
    Discard tiny noise detections, cap total detections to max_objects,
    and sort objects from left to right (x1 ascending).
    """
    img_area = img_w * img_h
    valid = []

    for item in boxes_with_conf:
        b = item["box"]
        w = b[2] - b[0]
        h = b[3] - b[1]
        area = w * h

        if w >= min_size and h >= min_size and (area / img_area) >= min_area_ratio:
            valid.append(item)

    if not valid:
        return []

    # If > max_objects, sort by confidence / area to keep the most prominent
    if len(valid) > max_objects:
        valid = sorted(valid, key=lambda x: (x.get("conf", 0.0), (x["box"][2] - x["box"][0]) * (x["box"][3] - x["box"][1])), reverse=True)[:max_objects]

    # Sort left-to-right (x1 ascending) for intuitive user inspection (#1, #2, #3...)
    valid = sorted(valid, key=lambda x: x["box"][0])
    return valid


# =====================================================================
# MODEL LOADING & CACHING
# =====================================================================
def _get_st_cache_decorator():
    """Dynamically acquire Streamlit cache_resource decorator if inside Streamlit."""
    try:
        import streamlit as st
        return st.cache_resource(show_spinner=False)
    except Exception:
        return lambda f: f


@_get_st_cache_decorator()
def load_detector(model_name: str = "yolov8n.pt"):
    """
    Load YOLO detection model.
    Uses cached resource to load once across sessions.
    Returns None if Ultralytics is unavailable or model loading fails.
    """
    try:
        from ultralytics import YOLO
        model = YOLO(model_name)
        # Warmup model once with a dummy 32x32 image
        import numpy as np
        dummy = np.zeros((32, 32, 3), dtype=np.uint8)
        model.predict(source=dummy, verbose=False, device="cpu")
        return model
    except Exception as exc:
        logger.warning("Failed to load YOLO model %s: %s", model_name, exc)
        return None


# =====================================================================
# OBJECT DETECTION PIPELINE
# =====================================================================
def detect_objects(
    image: Image.Image,
    model: Optional[Any] = None,
    conf_threshold: float = 0.18,
    max_objects: int = 6,
    padding_ratio: float = 0.10,
) -> List[Dict[str, Any]]:
    """
    Detect individual objects in an image.

    Args:
        image: PIL.Image in RGB mode.
        model: Ultralytics YOLO model instance (optional; loaded automatically if None).
        conf_threshold: Low detection confidence threshold for class-agnostic discovery.
        max_objects: Maximum number of objects to return.
        padding_ratio: Fraction of width/height to pad around each box.

    Returns:
        List of dicts: [{"box": [x1, y1, x2, y2], "crop": PIL.Image, "conf": float}, ...]
        Returns empty list [] if detector is unavailable or no objects are found.
    """
    if image is None:
        return []

    img_w, img_h = image.size
    if img_w < 10 or img_h < 10:
        return []

    # Acquire model
    if model is None:
        model = load_detector()
    if model is None:
        return []

    try:
        # Resize for fast CPU inference (longest side 640px)
        scale = 1.0
        max_dim = max(img_w, img_h)
        if max_dim > 640:
            scale = 640.0 / max_dim
            infer_w = int(img_w * scale)
            infer_h = int(img_h * scale)
            infer_img = image.resize((infer_w, infer_h), Image.Resampling.BILINEAR)
        else:
            infer_img = image

        # Class-agnostic prediction
        results = model.predict(
            source=infer_img,
            conf=conf_threshold,
            iou=0.45,
            agnostic_nms=True,
            verbose=False,
            device="cpu",
        )

        raw_boxes = []
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes_data = results[0].boxes
            for i in range(len(boxes_data)):
                xyxy = boxes_data.xyxy[i].cpu().numpy().tolist()
                conf = float(boxes_data.conf[i].cpu().item()) if boxes_data.conf is not None else 1.0

                # Scale coordinates back to original image resolution
                orig_x1 = int(round(xyxy[0] / scale))
                orig_y1 = int(round(xyxy[1] / scale))
                orig_x2 = int(round(xyxy[2] / scale))
                orig_y2 = int(round(xyxy[3] / scale))

                clamped = clamp_box([orig_x1, orig_y1, orig_x2, orig_y2], img_w, img_h)
                raw_boxes.append({
                    "box": clamped,
                    "conf": conf,
                })

        # Overlap merging to avoid splitting one object into multiple parts
        merged_boxes = merge_overlapping_boxes(raw_boxes, iou_thresh=0.60, containment_thresh=0.85)

        # Filter small noise boxes & cap
        filtered = filter_and_cap_boxes(merged_boxes, img_w, img_h, max_objects=max_objects)

        # Extract padded crops from original high-res image
        crops = []
        for det in filtered:
            padded = pad_box(det["box"], img_w, img_h, padding_ratio=padding_ratio)
            crop_img = image.crop((padded[0], padded[1], padded[2], padded[3]))
            crops.append({
                "box": padded,
                "crop": crop_img,
                "conf": det["conf"],
            })

        return crops

    except Exception as exc:
        logger.warning("YOLO detection error: %s", exc)
        return []


# =====================================================================
# TWO-STAGE CLASSIFICATION & FALLBACK
# =====================================================================
def classify_objects(
    image: Image.Image,
    classifier: Any,
    candidate_labels: List[str],
    detector_model: Optional[Any] = None,
    conf_threshold: float = 0.18,
    max_objects: int = 6,
    multi_object_enabled: bool = True,
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Run detection and classification.

    Args:
        image: PIL Image to analyze.
        classifier: Existing CLIP pipeline callable.
        candidate_labels: List of candidate waste labels.
        detector_model: Optional pre-loaded YOLO model.
        conf_threshold: Detection confidence threshold.
        max_objects: Maximum number of objects.
        multi_object_enabled: If False, skips YOLO and classifies whole image.

    Returns:
        (detections, is_fallback):
            detections: List of dicts, each containing:
                - "box": [x1, y1, x2, y2]
                - "crop": PIL Image of the object
                - "best_label": top predicted label string
                - "best_score": similarity score float
                - "results": list of top 3 predictions
                - "detector_conf": YOLO confidence float
            is_fallback: True if whole-image fallback was used (0 or 1 objects, or YOLO disabled/failed).
    """
    img_w, img_h = image.size
    detected_objects: List[Dict[str, Any]] = []

    if multi_object_enabled:
        detected_objects = detect_objects(
            image,
            model=detector_model,
            conf_threshold=conf_threshold,
            max_objects=max_objects,
        )

    # Fallback to whole-image if:
    # 1. Multi-object toggle is OFF, OR
    # 2. YOLO returned 0 objects, OR
    # 3. YOLO returned only 1 object (treat whole image as the primary item)
    if not detected_objects or len(detected_objects) <= 1:
        # If 1 object was found and covers most of the image, we can use either; whole-image gives full context
        predictions = classifier(image, candidate_labels=candidate_labels)[:3]
        fallback_item = {
            "box": [0, 0, img_w, img_h],
            "crop": image,
            "best_label": predictions[0]["label"],
            "best_score": predictions[0]["score"],
            "results": predictions,
            "detector_conf": detected_objects[0]["conf"] if detected_objects else 1.0,
        }
        return [fallback_item], True

    # Multi-object path: classify each detected crop individually
    classified: List[Dict[str, Any]] = []
    for det in detected_objects:
        try:
            preds = classifier(det["crop"], candidate_labels=candidate_labels)[:3]
            classified.append({
                "box": det["box"],
                "crop": det["crop"],
                "best_label": preds[0]["label"],
                "best_score": preds[0]["score"],
                "results": preds,
                "detector_conf": det["conf"],
            })
        except Exception as exc:
            logger.warning("Classification failed for crop %s: %s", det["box"], exc)

    if not classified:
        # Extreme fallback if all crop classifications failed
        predictions = classifier(image, candidate_labels=candidate_labels)[:3]
        return [{
            "box": [0, 0, img_w, img_h],
            "crop": image,
            "best_label": predictions[0]["label"],
            "best_score": predictions[0]["score"],
            "results": predictions,
            "detector_conf": 1.0,
        }], True

    return classified, False


# =====================================================================
# ANNOTATION & DRAWING
# =====================================================================
def draw_bounding_boxes(
    image: Image.Image,
    detections: List[Dict[str, Any]],
    show_labels: bool = True,
) -> Image.Image:
    """
    Draw clean, high-contrast bounding boxes with numbered badges (#1, #2...)
    and optional predicted labels on the image.
    """
    annotated = image.copy().convert("RGBA")
    draw = ImageDraw.Draw(annotated, "RGBA")
    overlay = Image.new("RGBA", annotated.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    # Try to load a clean truetype font, or fall back to default
    try:
        font = ImageFont.load_default(size=14)
    except Exception:
        font = ImageFont.load_default()

    for idx, det in enumerate(detections, start=1):
        box = det.get("box", [0, 0, image.width, image.height])
        color = BADGE_COLORS[(idx - 1) % len(BADGE_COLORS)]
        x1, y1, x2, y2 = box

        # 1. Semi-transparent box fill for visual separation
        # Convert hex color to rgba
        hex_c = color.lstrip("#")
        r, g, b = tuple(int(hex_c[i:i+2], 16) for i in (0, 2, 4))
        overlay_draw.rectangle([x1, y1, x2, y2], fill=(r, g, b, 28))

        # 2. Crisp outer boundary line
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # 3. Numbered badge tag at top-left of box
        label_text = f"#{idx}"
        if show_labels and det.get("best_label"):
            label_text = f"#{idx} {det['best_label'].title()}"

        # Measure text width
        try:
            bbox = font.getbbox(label_text)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
        except Exception:
            tw = len(label_text) * 7
            th = 14

        tag_x1 = x1
        tag_y1 = max(0, y1 - th - 8)
        tag_x2 = tag_x1 + tw + 12
        tag_y2 = tag_y1 + th + 8

        # Badge pill
        draw.rectangle([tag_x1, tag_y1, tag_x2, tag_y2], fill=(6, 36, 26, 235))
        draw.rectangle([tag_x1, tag_y1, tag_x2, tag_y2], outline=color, width=1)
        draw.text((tag_x1 + 6, tag_y1 + 4), label_text, fill=(228, 251, 194, 255), font=font)

    # Composite overlay
    result = Image.alpha_composite(annotated, overlay).convert("RGB")
    return result
