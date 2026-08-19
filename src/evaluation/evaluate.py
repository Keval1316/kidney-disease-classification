"""
Evaluate the trained model on the held-out test set.
Produces confusion_matrix.png, classification_report.json, metrics.json.

Usage:
    python -m src.evaluation.evaluate
"""
import json
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    accuracy_score,
)

from src.preprocessing.preprocessing import build_datasets
from src.utils.logger import get_logger

logger = get_logger(__name__)

CONFIG_PATH = "config/config.yaml"
PARAMS_PATH = "params.yaml"


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Test Set")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"Confusion matrix saved to {out_path}")


def main():
    config = load_yaml(CONFIG_PATH)

    model_path = Path(config["paths"]["model_dir"]) / "best_model.keras"
    logger.info(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path)

    logger.info("Building test dataset...")
    _, _, test_ds, class_index = build_datasets(
        splits_path="data/processed/splits.json",
        config_path=CONFIG_PATH,
        params_path=PARAMS_PATH,
    )
    # index -> name, ordered by index value
    idx_to_name = {v: k for k, v in class_index.items()}
    class_names = [idx_to_name[i] for i in range(len(idx_to_name))]

    logger.info("Running predictions on test set...")
    y_true = []
    y_pred = []
    y_prob = []

    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        preds = np.argmax(probs, axis=1)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(preds.tolist())
        y_prob.extend(probs.tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Overall accuracy
    acc = accuracy_score(y_true, y_pred)

    # Macro and weighted precision/recall/F1
    # Macro: unweighted mean across classes — treats Stone (minority) equally
    # to Normal (majority). Weighted: accounts for class support/frequency.
    # We report both, but MACRO is the primary metric here since this is a
    # medical-imaging problem where missing the minority class (Stone) matters
    # just as much as missing the majority class (Normal) — a model that's
    # great on Normal/Cyst but bad on Stone would score well on weighted
    # metrics while being clinically unreliable on the rarer condition.
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "test_accuracy": float(acc),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "num_test_samples": int(len(y_true)),
    }

    reports_dir = Path(config["paths"]["reports_dir"])
    metrics_dir = reports_dir / "metrics"
    figures_dir = reports_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    with open(metrics_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_dir / 'metrics.json'}")

    with open(metrics_dir / "classification_report.json", "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Classification report saved to {metrics_dir / 'classification_report.json'}")

    plot_confusion_matrix(cm, class_names, figures_dir / "confusion_matrix.png")

    # also save raw predictions for Phase 11 error analysis
    predictions_out = {
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "y_prob": y_prob,
        "class_names": class_names,
    }
    with open(metrics_dir / "test_predictions.json", "w") as f:
        json.dump(predictions_out, f)
    logger.info("Raw test predictions saved for error analysis (Phase 11).")

    logger.info(f"Test accuracy: {acc:.4f}")
    logger.info(f"Macro   — precision: {precision_macro:.4f}, recall: {recall_macro:.4f}, F1: {f1_macro:.4f}")
    logger.info(f"Weighted— precision: {precision_weighted:.4f}, recall: {recall_weighted:.4f}, F1: {f1_weighted:.4f}")
    logger.info("Per-class breakdown:")
    for cls in class_names:
        cls_metrics = report[cls]
        logger.info(
            f"  {cls:<10} precision={cls_metrics['precision']:.3f} "
            f"recall={cls_metrics['recall']:.3f} f1={cls_metrics['f1-score']:.3f} "
            f"support={int(cls_metrics['support'])}"
        )


if __name__ == "__main__":
    main()  