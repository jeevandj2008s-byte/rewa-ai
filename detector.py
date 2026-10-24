"""detector.py - Multi-Object Waste Detection for ReWA.AI

Two-stage computer vision pipeline:
1. Stage 1: Pre-trained YOLOv8 waste detection (Ultralytics) to isolate individual objects,
   predict spatial coordinates (bounding boxes), and identify initial waste classes.
2. Stage 2: Target-specific CLIP zero-shot classification on each object crop to refine
   fine-grained waste categories (e.g., rigid bottle vs soft wrapper).

Model Details:
- Model: `kendrickfff/waste-classification-yolov8-ken`
- Architecture: YOLOv8 Nano (`yolov8n`), PyTorch weights
- License: CC-BY-4.0 (Model weights); Ultralytics AGPL-3.0
- Classes (12): battery, biological, brown-glass, cardboard, clothes,
  green-glass, metal, paper, plastic, shoes, trash, white-glass.
- Class mapping: defined in data/yolo_class_map.json.

Streamlit Cloud & CPU Optimizations:
- Aspect-ratio preserving resizing (longest side 640px) for fast CPU inference (<100ms).
- Coordinate clamping, adaptive padding, noise filtering, containment merging.
- High-contrast visual annotation with numbered badge pills (#1, #2, ...).
- Automatic Hugging Face Hub download & disk caching.
- Graceful fallback to whole-image CLIP classification when YOLO detects 0 items or fails.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# Repository root and model directory
REPO_ROOT = Path(__file__).resolve().parent
MODELS_DIR = REPO_ROOT / "models"
DATA_DIR = REPO_ROOT / "data"

# Default pre-trained waste model configuration
DEFAULT_WASTE_MODEL_REPO = "kendrickfff/waste-classification-yolov8-ken"
DEFAULT_WASTE_MODEL_FILE = "yolov8n-waste-12cls-best.pt"
LOCAL_WASTE_MODEL_PATH = MODELS_DIR / DEFAULT_WASTE_MODEL_FILE
CLASS_MAP_PATH = DATA_DIR / "yolo_class_map.json"

# High-contrast color palette for visual badges (#1, #2, #3, ...)
BADGE_COLORS = [
    "#22C55E",  # Emerald Green (#1)
    "#F59E0B",  # Amber (#2)
    "#0EA5E9",  # Sky Blue (#3)
    "#EC4899",  # Pink (#4)
    "#8B5CF6",  # Violet (#5)
    "#14B8A6",  # Teal (#6)
]

# In-memory cache for class mapping
_CACHED_CLASS_MAP: Optional[Dict[str, Any]] = None


# =====================================================================
# CLASS MAPPING & METADATA HELPERS
# =====================================================================
def load_class_map() -> Dict[str, Any]:
    """Load YOLO waste class mapping from data/yolo_class_map.json."""
    global _CACHED_CLASS_MAP
    if _CACHED_CLASS_MAP is not None:
        return _CACHED_CLASS_MAP

    if CLASS_MAP_PATH.exists():
        try:
            with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
                _CACHED_CLASS_MAP = json.load(f)
                return _CACHED_CLASS_MAP
        except Exception as exc:
            logger.warning("Failed to load yolo_class_map.json: %s", exc)

    # Fallback default mapping if file is missing
    _CACHED_CLASS_MAP = {
        "classes": {
            "battery": {"canonical_bin": "Hazardous", "default_label": "battery", "candidate_labels": ["battery"]},
            "biological": {"canonical_bin": "Biodegradable", "default_label": "food waste", "candidate_labels": ["food waste", "fruit peel"]},
            "brown-glass": {"canonical_bin": "Glass", "default_label": "glass bottle", "candidate_labels": ["glass bottle", "glass jar"]},
            "cardboard": {"canonical_bin": "Paper", "default_label": "cardboard box", "candidate_labels": ["cardboard box", "paper sheet", "newspaper"]},
            "clothes": {"canonical_bin": None, "default_label": "other waste", "candidate_labels": ["other waste"]},
            "green-glass": {"canonical_bin": "Glass", "default_label": "glass bottle", "candidate_labels": ["glass bottle", "glass jar"]},
            "metal": {"canonical_bin": "Metal", "default_label": "metal can", "candidate_labels": ["metal can", "aerosol can", "paint can"]},
            "paper": {"canonical_bin": "Paper", "default_label": "paper sheet", "candidate_labels": ["paper sheet", "newspaper", "cardboard box"]},
            "plastic": {"canonical_bin": "Plastic", "default_label": "plastic bottle", "candidate_labels": ["plastic bottle", "plastic wrapper", "plastic bag"]},
            "shoes": {"canonical_bin": None, "default_label": "other waste", "candidate_labels": ["other waste"]},
            "trash": {"canonical_bin": None, "default_label": "other waste", "candidate_labels": ["other waste"]},
            "white-glass": {"canonical_bin": "Glass", "default_label": "glass jar", "candidate_labels": ["glass jar", "glass bottle"]},
        }
    }
    return _CACHED_CLASS_MAP


def download_waste_model() -> Optional[str]:
    """
    Ensure the pre-trained waste YOLO model weights are available locally.
    Checks models/ directory first, then downloads via huggingface_hub if needed.
    Returns path string to the weights file, or None if unavailable.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Check local models/ directory
    if LOCAL_WASTE_MODEL_PATH.exists() and LOCAL_WASTE_MODEL_PATH.stat().st_size > 1000:
        return str(LOCAL_WASTE_MODEL_PATH)

    # 2. Check root directory fallback
    root_model = REPO_ROOT / DEFAULT_WASTE_MODEL_FILE
    if root_model.exists() and root_model.stat().st_size > 1000:
        return str(root_model)

    # 3. Download from Hugging Face Hub
    try:
        from huggingface_hub import hf_hub_download
        logger.info(
            "Downloading pre-trained waste detector (%s) from Hugging Face...",
            DEFAULT_WASTE_MODEL_REPO,
        )
        downloaded_path = hf_hub_download(
            repo_id=DEFAULT_WASTE_MODEL_REPO,
            filename=DEFAULT_WASTE_MODEL_FILE,
            local_dir=str(MODELS_DIR),
        )
        if os.path.exists(downloaded_path):
            return str(downloaded_path)
    except Exception as exc:
        logger.warning(
            "Could not download waste model from Hugging Face Hub (%s): %s. "
            "Falling back to local models or yolov8n.pt if present.",
            DEFAULT_WASTE_MODEL_REPO,
            exc,
        )

    # 4. Fallback to generic yolov8n.pt if present locally
    coco_local = REPO_ROOT / "yolov8n.pt"
    if coco_local.exists():
        return str(coco_local)

    return None


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
        curr_cls = detections[i].get("cls_name", "")
        curr_cls_id = detections[i].get("cls_id")

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
                other_conf = float(detections[j].get("conf", 1.0))
                if other_conf > curr_conf:
                    curr_conf = other_conf
                    curr_cls = detections[j].get("cls_name", curr_cls)
                    curr_cls_id = detections[j].get("cls_id", curr_cls_id)
                consumed[j] = True

        consumed[i] = True
        merged.append({
            "box": curr_box,
            "conf": curr_conf,
            "cls_name": curr_cls,
            "cls_id": curr_cls_id,
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
        valid = sorted(
            valid,
            key=lambda x: (x.get("conf", 0.0), (x["box"][2] - x["box"][0]) * (x["box"][3] - x["box"][1])),
            reverse=True,
        )[:max_objects]

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
def load_detector(model_name: Optional[str] = None):
    """
    Load YOLO detection model.
    Prioritizes pre-trained waste classification model (kendrickfff/waste-classification-yolov8-ken).
    Uses Streamlit cache_resource to load once across sessions.
    Returns None if Ultralytics is unavailable or model loading fails.
    """
    try:
        from ultralytics import YOLO

        target_path: Optional[str] = model_name
        if target_path is None or target_path in ("yolov8n.pt", DEFAULT_WASTE_MODEL_FILE):
            # Resolve waste model weights
            waste_path = download_waste_model()
            if waste_path:
                target_path = waste_path
            else:
                target_path = "yolov8n.pt"

        logger.info("Loading YOLO detector from: %s", target_path)
        model = YOLO(target_path)

        # Warmup model once with a dummy 32x32 image on CPU
        import numpy as np
        dummy = np.zeros((32, 32, 3), dtype=np.uint8)
        model.predict(source=dummy, verbose=False, device="cpu")
        return model
    except Exception as exc:
        logger.warning("Failed to load YOLO model %s: %s", model_name, exc)
        return None


def get_detector_info(model: Optional[Any] = None) -> Dict[str, Any]:
    """Return inspection metadata for the active YOLO model."""
    if model is None:
        model = load_detector()
    if model is None:
        return {"loaded": False, "model_name": None, "classes": []}

    names = getattr(model, "names", {})
    if isinstance(names, dict):
        class_list = [names[k] for k in sorted(names.keys())]
    else:
        class_list = list(names)

    is_waste_model = any(c in class_list for c in ("cardboard", "biological", "battery", "white-glass"))
    return {
        "loaded": True,
        "is_waste_model": is_waste_model,
        "total_classes": len(class_list),
        "classes": class_list,
        "model_file": getattr(model, "model_name", "yolo"),
    }


# =====================================================================
# OBJECT DETECTION PIPELINE
# =====================================================================
def detect_objects(
    image: Image.Image,
    model: Optional[Any] = None,
    conf_threshold: float = 0.25,
    max_objects: int = 6,
    padding_ratio: float = 0.10,
) -> List[Dict[str, Any]]:
    """
    Detect individual objects in an image.

    Args:
        image: PIL.Image in RGB mode.
        model: Ultralytics YOLO model instance (optional; loaded automatically if None).
        conf_threshold: Detection confidence threshold (default 0.25).
        max_objects: Maximum number of objects to return.
        padding_ratio: Fraction of width/height to pad around each box.

    Returns:
        List of dicts: [
            {"box": [x1, y1, x2, y2], "crop": PIL.Image, "conf": float, "cls_name": str, "cls_id": int},
            ...
        ]
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

        # Predict with YOLO
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

                # Extract detected class label
                cls_id = None
                cls_name = ""
                if hasattr(boxes_data, "cls") and boxes_data.cls is not None:
                    cls_id = int(boxes_data.cls[i].cpu().item())
                    if hasattr(model, "names") and isinstance(model.names, dict):
                        cls_name = model.names.get(cls_id, "")
                    elif hasattr(model, "names") and isinstance(model.names, list) and cls_id < len(model.names):
                        cls_name = model.names[cls_id]

                # Scale coordinates back to original image resolution
                orig_x1 = int(round(xyxy[0] / scale))
                orig_y1 = int(round(xyxy[1] / scale))
                orig_x2 = int(round(xyxy[2] / scale))
                orig_y2 = int(round(xyxy[3] / scale))

                clamped = clamp_box([orig_x1, orig_y1, orig_x2, orig_y2], img_w, img_h)
                raw_boxes.append({
                    "box": clamped,
                    "conf": conf,
                    "cls_name": cls_name,
                    "cls_id": cls_id,
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
                "cls_name": det.get("cls_name", ""),
                "cls_id": det.get("cls_id"),
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
    conf_threshold: float = 0.25,
    max_objects: int = 6,
    multi_object_enabled: bool = True,
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Run detection and classification.

    Args:
        image: PIL Image to analyze.
        classifier: Existing CLIP pipeline callable.
        candidate_labels: Full list of candidate waste labels.
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
                - "yolo_class": YOLO detected class string
                - "canonical_bin": canonical bin if known
            is_fallback: True if whole-image fallback was used (0 or 1 objects, or YOLO disabled/failed).
    """
    img_w, img_h = image.size
    detected_objects: List[Dict[str, Any]] = []
    class_map = load_class_map().get("classes", {})

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
    # 3. YOLO returned only 1 object (treat whole image as the primary item for maximum context)
    if not detected_objects or len(detected_objects) <= 1:
        predictions = classifier(image, candidate_labels=candidate_labels)[:3]
        fallback_item = {
            "box": [0, 0, img_w, img_h],
            "crop": image,
            "best_label": predictions[0]["label"],
            "best_score": predictions[0]["score"],
            "results": predictions,
            "detector_conf": detected_objects[0]["conf"] if detected_objects else 1.0,
            "yolo_class": detected_objects[0].get("cls_name", "") if detected_objects else "",
            "canonical_bin": None,
        }
        return [fallback_item], True

    # Multi-object path: classify each detected crop individually
    classified: List[Dict[str, Any]] = []
    for det in detected_objects:
        det_cls = det.get("cls_name", "")
        # Resolve target candidate labels for CLIP refinement
        if det_cls and det_cls in class_map:
            cls_meta = class_map[det_cls]
            target_labels = cls_meta.get("candidate_labels", candidate_labels)
            canonical_bin = cls_meta.get("canonical_bin")
            default_label = cls_meta.get("default_label", "other waste")
        else:
            target_labels = candidate_labels
            canonical_bin = None
            default_label = "other waste"

        try:
            preds = classifier(det["crop"], candidate_labels=target_labels)[:3]
            classified.append({
                "box": det["box"],
                "crop": det["crop"],
                "best_label": preds[0]["label"],
                "best_score": preds[0]["score"],
                "results": preds,
                "detector_conf": det["conf"],
                "yolo_class": det_cls,
                "canonical_bin": canonical_bin,
            })
        except Exception as exc:
            logger.warning("Classification failed for crop %s: %s", det["box"], exc)
            # Use YOLO detection info directly if CLIP crop classification fails
            classified.append({
                "box": det["box"],
                "crop": det["crop"],
                "best_label": default_label,
                "best_score": det["conf"],
                "results": [{"label": default_label, "score": det["conf"]}],
                "detector_conf": det["conf"],
                "yolo_class": det_cls,
                "canonical_bin": canonical_bin,
            })

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
            "yolo_class": "",
            "canonical_bin": None,
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
        hex_c = color.lstrip("#")
        r, g, b = tuple(int(hex_c[i:i+2], 16) for i in (0, 2, 4))
        overlay_draw.rectangle([x1, y1, x2, y2], fill=(r, g, b, 28))

        # 2. Crisp outer boundary line
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # 3. Numbered badge tag at top-left of box
        label_text = f"#{idx}"
        display_label = det.get("best_label") or det.get("yolo_class")
        if show_labels and display_label:
            label_text = f"#{idx} {display_label.title()}"

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
