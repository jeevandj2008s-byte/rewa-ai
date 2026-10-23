#!/usr/bin/env python3
"""evaluate_model.py - Measure zero-shot CLIP classification performance.

Evaluates ReWA.AI's waste classifier against a folder of labeled images.
Expected folder structure:
    eval_data/
    ├── <label_name>/
    │   ├── image1.jpg
    │   └── image2.png
    └── <another_label>/
        └── image3.jpg

If eval_data/ is empty or missing, this script exits gracefully with instructions.
NO fake data or fabricated numbers are ever generated.
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from PIL import Image

# Import single source of truth for candidate labels
try:
    from bin_rules import CANDIDATE_LABELS
except ImportError:
    vscode_path = Path(__file__).resolve().parent / ".vscode"
    if str(vscode_path) not in sys.path:
        sys.path.insert(0, str(vscode_path))
    from bin_rules import CANDIDATE_LABELS

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def find_eval_images(eval_dir):
    """Scan eval_dir for labeled images organized by folder name.

    Returns a list of dicts: [{"path": Path, "true_label": str}]
    """
    eval_path = Path(eval_dir)
    if not eval_path.exists() or not eval_path.is_dir():
        return []

    samples = []
    for item in eval_path.iterdir():
        if item.is_dir():
            folder_label = item.name.strip().lower()
            for file_path in item.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
                    samples.append({
                        "path": file_path,
                        "true_label": folder_label,
                    })
    return samples


def run_evaluation(eval_dir="eval_data", output_dir=".", model_name="openai/clip-vit-large-patch14"):
    samples = find_eval_images(eval_dir)

    if not samples:
        print("=" * 72)
        print("ReWA.AI — Waste Classifier Evaluation")
        print("=" * 72)
        print(f"No labeled images found in '{eval_dir}/'.")
        print("\nTo evaluate real model performance:")
        print(f"  1. Create class folders inside '{eval_dir}/' matching candidate labels.")
        print("     Example:")
        print(f"       {eval_dir}/plastic bottle/sample_1.jpg")
        print(f"       {eval_dir}/cardboard box/sample_2.jpg")
        print(f"       {eval_dir}/metal can/sample_3.png")
        print("  2. Recommended: Add ~5-10 real photos per common waste category.")
        print("\nAvailable candidate labels:")
        for i, lbl in enumerate(CANDIDATE_LABELS, 1):
            print(f"    {i:2d}. {lbl}")
        print("\n  3. Re-run: python evaluate_model.py")
        print("=" * 72)
        return 0

    print(f"[ReWA.AI] Found {len(samples)} image(s) across folders in '{eval_dir}/'.")
    print(f"[ReWA.AI] Loading zero-shot classifier: {model_name}...")

    from transformers import pipeline
    classifier = pipeline("zero-shot-image-classification", model=model_name)

    print(f"[ReWA.AI] Evaluating against {len(CANDIDATE_LABELS)} candidate labels...")

    results = []
    per_label_total = Counter()
    per_label_top1 = Counter()
    per_label_top3 = Counter()
    confusion_pairs = Counter()

    for idx, sample in enumerate(samples, 1):
        file_path = sample["path"]
        true_label = sample["true_label"]
        per_label_total[true_label] += 1

        try:
            with Image.open(file_path) as raw_img:
                img = raw_img.convert("RGB")
                preds = classifier(img, candidate_labels=CANDIDATE_LABELS)[:3]
        except Exception as error:
            print(f"  [ERROR] Failed to process {file_path}: {error}")
            continue

        top1_pred = preds[0]["label"].lower()
        top1_score = preds[0]["score"]
        top3_preds = [p["label"].lower() for p in preds]

        is_top1 = (top1_pred == true_label)
        is_top3 = (true_label in top3_preds)

        if is_top1:
            per_label_top1[true_label] += 1
        else:
            confusion_pairs[(true_label, top1_pred)] += 1

        if is_top3:
            per_label_top3[true_label] += 1

        results.append({
            "filename": file_path.name,
            "path": str(file_path),
            "true_label": true_label,
            "top1_prediction": top1_pred,
            "top1_similarity_score": round(top1_score, 4),
            "top1_correct": is_top1,
            "top3_predictions": top3_preds,
            "top3_correct": is_top3,
        })

        if idx % 5 == 0 or idx == len(samples):
            print(f"  Processed {idx}/{len(samples)} images...")

    total_evaluated = len(results)
    if total_evaluated == 0:
        print("[ReWA.AI] No images were successfully processed.")
        return 1

    overall_top1_acc = sum(per_label_top1.values()) / total_evaluated
    overall_top3_acc = sum(per_label_top3.values()) / total_evaluated

    per_label_report = {}
    for label, count in per_label_total.items():
        top1_c = per_label_top1[label]
        top3_c = per_label_top3[label]
        per_label_report[label] = {
            "samples": count,
            "top1_correct": top1_c,
            "top1_accuracy": round(top1_c / count, 4),
            "top3_correct": top3_c,
            "top3_accuracy": round(top3_c / count, 4),
        }

    common_mistakes = [
        {"true_label": true_lbl, "predicted_label": pred_lbl, "occurrences": count}
        for (true_lbl, pred_lbl), count in confusion_pairs.most_common(10)
    ]

    report = {
        "timestamp": datetime.now().isoformat(),
        "model_name": model_name,
        "dataset_path": str(Path(eval_dir).resolve()),
        "total_images_evaluated": total_evaluated,
        "overall_top1_accuracy": round(overall_top1_acc, 4),
        "overall_top3_accuracy": round(overall_top3_acc, 4),
        "per_label_accuracy": per_label_report,
        "most_common_mistakes": common_mistakes,
    }

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. JSON Report
    json_path = out_path / "eval_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[ReWA.AI] Saved JSON report to: {json_path}")

    # Also save a copy inside .vscode so Streamlit finds it easily
    vscode_json_path = Path(".vscode") / "eval_report.json"
    if vscode_json_path.parent.exists():
        try:
            with open(vscode_json_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
        except Exception:
            pass

    # 2. Markdown Report
    md_path = out_path / "eval_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# ReWA.AI Model Evaluation Report\n\n")
        f.write(f"- **Evaluated At:** {report['timestamp']}\n")
        f.write(f"- **Model:** `{model_name}`\n")
        f.write(f"- **Total Samples:** {total_evaluated}\n")
        f.write(f"- **Overall Top-1 Accuracy:** {overall_top1_acc:.1%}\n")
        f.write(f"- **Overall Top-3 Accuracy:** {overall_top3_acc:.1%}\n\n")

        f.write("## Per-Label Accuracy\n\n")
        f.write("| Label | Samples | Top-1 Accuracy | Top-3 Accuracy |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        for lbl, stats in sorted(per_label_report.items()):
            f.write(f"| {lbl} | {stats['samples']} | {stats['top1_accuracy']:.1%} | {stats['top3_accuracy']:.1%} |\n")

        if common_mistakes:
            f.write("\n## Common Confusion Pairs (Mistakes)\n\n")
            f.write("| True Label | Predicted As | Count |\n")
            f.write("| :--- | :--- | :---: |\n")
            for m in common_mistakes:
                f.write(f"| {m['true_label']} | {m['predicted_label']} | {m['occurrences']} |\n")
    print(f"[ReWA.AI] Saved Markdown report to: {md_path}")

    # 3. CSV Details
    csv_path = out_path / "eval_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filename", "true_label", "top1_prediction", "top1_similarity_score",
            "top1_correct", "top3_predictions", "top3_correct", "path"
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "filename": r["filename"],
                "true_label": r["true_label"],
                "top1_prediction": r["top1_prediction"],
                "top1_similarity_score": r["top1_similarity_score"],
                "top1_correct": r["top1_correct"],
                "top3_predictions": ", ".join(r["top3_predictions"]),
                "top3_correct": r["top3_correct"],
                "path": r["path"],
            })
    print(f"[ReWA.AI] Saved CSV sample details to: {csv_path}")

    print("\n" + "=" * 72)
    print(f"Overall Top-1 Accuracy: {overall_top1_acc:.1%}")
    print(f"Overall Top-3 Accuracy: {overall_top3_acc:.1%}")
    print("=" * 72)
    return 0


def main():
    parser = argparse.ArgumentParser(description="Evaluate ReWA.AI CLIP Waste Classifier.")
    parser.add_argument("--data-dir", default="eval_data", help="Path to labeled eval images folder")
    parser.add_argument("--output-dir", default=".", help="Directory to save evaluation reports")
    args = parser.parse_args()

    sys.exit(run_evaluation(eval_dir=args.data_dir, output_dir=args.output_dir))


if __name__ == "__main__":
    main()

