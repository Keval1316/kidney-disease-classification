"""
Error analysis using saved test predictions from Phase 10.
Identifies correct/incorrect predictions, lowest-confidence predictions,
and highest-confidence wrong predictions. Saves representative image
examples (not all — a bounded sample) for visual review.

Usage:
    python -m src.evaluation.error_analysis
"""
import json
import shutil
from pathlib import Path

import numpy as np
import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)

CONFIG_PATH = "config/config.yaml"
N_EXAMPLES_PER_CATEGORY = 8  # bounded sample, not everything


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_test_filepaths(splits_path: str) -> list:
    """The test split order in splits.json must match the order predictions
    were generated in (Phase 10 iterates test_ds without shuffling — verified
    in preprocessing.py's _make_dataset(shuffle=False) for test). Returns
    list of (filepath, class_name) in that same order."""
    with open(splits_path, "r") as f:
        splits = json.load(f)
    return [tuple(p) for p in splits["test"]]


def main():
    config = load_yaml(CONFIG_PATH)
    reports_dir = Path(config["paths"]["reports_dir"])
    metrics_dir = reports_dir / "metrics"
    figures_dir = reports_dir / "figures"

    pred_path = metrics_dir / "test_predictions.json"
    if not pred_path.exists():
        logger.error(f"{pred_path} not found — run `python -m src.evaluation.evaluate` first.")
        return

    with open(pred_path, "r") as f:
        preds = json.load(f)

    y_true = np.array(preds["y_true"])
    y_pred = np.array(preds["y_pred"])
    y_prob = np.array(preds["y_prob"])
    class_names = preds["class_names"]

    test_pairs = load_test_filepaths("data/processed/splits.json")
    if len(test_pairs) != len(y_true):
        logger.error(
            f"Mismatch: {len(test_pairs)} test files vs {len(y_true)} predictions. "
            "Splits may have changed since evaluate.py was last run — rerun evaluate.py."
        )
        return

    confidences = y_prob[np.arange(len(y_pred)), y_pred]
    correct_mask = y_true == y_pred

    # --- Category 1: highest-confidence WRONG predictions (most concerning) ---
    wrong_indices = np.where(~correct_mask)[0]
    wrong_by_confidence = wrong_indices[np.argsort(-confidences[wrong_indices])]
    top_wrong_confident = wrong_by_confidence[:N_EXAMPLES_PER_CATEGORY]

    # --- Category 2: lowest-confidence predictions overall (model unsure) ---
    lowest_confidence = np.argsort(confidences)[:N_EXAMPLES_PER_CATEGORY]

    # --- Category 3: sample of correct predictions (for contrast/sanity check) ---
    correct_indices = np.where(correct_mask)[0]
    sample_correct = correct_indices[:N_EXAMPLES_PER_CATEGORY] if len(correct_indices) > 0 else []

    def describe(indices, label):
        logger.info(f"\n--- {label} ---")
        entries = []
        for idx in indices:
            fp, true_cls = test_pairs[idx]
            pred_cls = class_names[y_pred[idx]]
            conf = float(confidences[idx])
            logger.info(
                f"  true={true_cls:<8} pred={pred_cls:<8} confidence={conf:.4f}  {Path(fp).name}"
            )
            entries.append({
                "filepath": fp,
                "true_class": true_cls,
                "predicted_class": pred_cls,
                "confidence": conf,
            })
        return entries

    top_wrong_entries = describe(top_wrong_confident, "Highest-confidence WRONG predictions")
    lowest_conf_entries = describe(lowest_confidence, "Lowest-confidence predictions (model unsure)")
    correct_entries = describe(sample_correct, "Sample of correct predictions")

    # Save a bounded set of copied example images for visual review
    examples_dir = figures_dir / "error_analysis_examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    def copy_examples(entries, subfolder):
        out_dir = examples_dir / subfolder
        out_dir.mkdir(parents=True, exist_ok=True)
        for e in entries:
            src = Path(e["filepath"])
            if src.exists():
                dst = out_dir / f"true-{e['true_class']}_pred-{e['predicted_class']}_conf-{e['confidence']:.2f}_{src.name}"
                shutil.copy(src, dst)
        logger.info(f"Copied {len(entries)} example images to {out_dir}")

    copy_examples(top_wrong_entries, "highest_confidence_wrong")
    copy_examples(lowest_conf_entries, "lowest_confidence")
    copy_examples(correct_entries, "sample_correct")

    # Save the summary as JSON too
    summary = {
        "total_test_samples": int(len(y_true)),
        "total_correct": int(correct_mask.sum()),
        "total_incorrect": int((~correct_mask).sum()),
        "highest_confidence_wrong": top_wrong_entries,
        "lowest_confidence": lowest_conf_entries,
        "sample_correct": correct_entries,
    }
    summary_path = metrics_dir / "error_analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nError analysis summary saved to {summary_path}")

    logger.info(
        f"\nTotal: {len(y_true)} test samples, "
        f"{correct_mask.sum()} correct ({correct_mask.mean()*100:.1f}%), "
        f"{(~correct_mask).sum()} incorrect"
    )


if __name__ == "__main__":
    main()