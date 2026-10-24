"""test_yolo.py - Verification and sanity testing for YOLO waste detection in ReWA.AI.

Runs end-to-end sanity tests:
1. Verifies model loading and prints architecture and model.names.
2. Validates class mapping in data/yolo_class_map.json.
3. Tests inference on a test image (or optional image path passed via command line).
4. Validates geometry helpers, bounding box annotation, and crop extraction.
"""

import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import detector
import bin_rules


def test_model_loading_and_classes():
    print("=" * 60)
    print("1. VERIFYING MODEL LOADING & CLASSES")
    print("=" * 60)
    model = detector.load_detector()
    if model is None:
        print("[FAIL] Could not load YOLO detector model!")
        return False

    info = detector.get_detector_info(model)
    print(f"Model loaded successfully: {info['loaded']}")
    print(f"Is waste-specific model:   {info['is_waste_model']}")
    print(f"Total detection classes:   {info['total_classes']}")
    print(f"Classes dictionary:        {getattr(model, 'names', {})}")

    if not info["is_waste_model"]:
        print("[WARN] Model loaded is not waste-specific (likely generic COCO fallback)!")
    else:
        print("[PASS] Verified genuine pre-trained waste detection model.")

    return True


def test_class_mapping_integrity():
    print("\n" + "=" * 60)
    print("2. VERIFYING CLASS MAPPING (data/yolo_class_map.json)")
    print("=" * 60)
    cmap = detector.load_class_map()
    classes = cmap.get("classes", {})
    print(f"Total mapped classes: {len(classes)}")

    for cls_name, meta in classes.items():
        bin_target = meta.get("canonical_bin")
        default_label = meta.get("default_label")
        print(f"  - {cls_name:12s} -> Bin: {str(bin_target):14s} | Label: {default_label}")
        if bin_target is not None:
            assert bin_target in bin_rules.CANONICAL_BINS, f"Invalid bin: {bin_target}"
        assert default_label in bin_rules.CANDIDATE_LABELS, f"Invalid default label: {default_label}"

    print("[PASS] 100% of classes map cleanly to canonical bins and candidate labels.")
    return True


def test_synthetic_detection_and_annotation(image_path: str = None):
    print("\n" + "=" * 60)
    print("3. TESTING DETECTION PIPELINE & VISUAL ANNOTATION")
    print("=" * 60)

    if image_path and os.path.exists(image_path):
        print(f"Loading provided image: {image_path}")
        img = Image.open(image_path).convert("RGB")
    else:
        print("Creating synthetic multi-object canvas for pipeline check...")
        img = Image.new("RGB", (640, 480), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)
        # Draw mock shapes (blue bottle shape, brown box shape)
        draw.rectangle([60, 80, 200, 400], fill=(40, 120, 220), outline=(10, 50, 150), width=3)
        draw.rectangle([340, 140, 560, 380], fill=(180, 130, 80), outline=(100, 70, 30), width=3)

    model = detector.load_detector()
    detections = detector.detect_objects(img, model=model, conf_threshold=0.20)
    print(f"Detections found: {len(detections)}")

    for idx, det in enumerate(detections, start=1):
        box = det["box"]
        conf = det.get("conf", 0.0)
        cls_name = det.get("cls_name", "unknown")
        crop_size = det["crop"].size
        print(f"  Object #{idx}: class='{cls_name}' conf={conf:.1%} box={box} crop_size={crop_size}")

    annotated = detector.draw_bounding_boxes(img, detections)
    output_path = REPO_ROOT / "models" / "test_detection_output.png"
    annotated.save(output_path)
    print(f"[PASS] Annotated output saved to: {output_path}")

    # Fallback test
    dummy_empty = Image.new("RGB", (200, 200), color=(255, 255, 255))
    fallback_res, is_fallback = detector.classify_objects(
        dummy_empty,
        classifier=lambda crop, candidate_labels: [{"label": "other waste", "score": 0.9}],
        candidate_labels=bin_rules.CANDIDATE_LABELS,
        detector_model=model,
        multi_object_enabled=True,
    )
    print(f"Fallback verification: is_fallback={is_fallback}, num_items={len(fallback_res)}")
    assert is_fallback is True, "Fallback should trigger on empty/no-object images"
    print("[PASS] Fallback mechanism verified.")

    return True


if __name__ == "__main__":
    cli_img = sys.argv[1] if len(sys.argv) > 1 else None
    s1 = test_model_loading_and_classes()
    s2 = test_class_mapping_integrity()
    s3 = test_synthetic_detection_and_annotation(cli_img)
    if s1 and s2 and s3:
        print("\n" + "=" * 60)
        print("ALL YOLO VERIFICATION CHECKS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
    else:
        sys.exit(1)

