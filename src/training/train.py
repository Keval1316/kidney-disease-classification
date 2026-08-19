"""
Full training pipeline: data -> model -> train (2-phase) -> save -> log to MLflow.

Two-phase transfer learning strategy
--------------------------------------
Phase 1 — Head warm-up (backbone frozen):
  - Only the new classification head trains.
  - Higher learning rate (warmup_lr, default 0.001) for fast convergence.
  - Runs for `warmup_epochs` epochs (default 8).
  - No early stopping in this phase so warm-up always completes.
  - This prevents the random head weights from destroying pretrained backbone
    features on the very first epoch.

Phase 2 — Fine-tuning (last N backbone layers unfrozen):
  - Unfreeze the last `trainable_layers` layers of EfficientNetB0.
  - Recompile with a low learning rate (learning_rate, default 5e-5).
  - Run for up to `epochs` more epochs with EarlyStopping.
  - The low LR ensures pretrained weights are updated carefully (not wiped).

Usage:
    python -m src.training.train
"""
import os
import sys
import json
import time
import subprocess
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from dotenv import load_dotenv
import mlflow
import mlflow.keras

from src.preprocessing.preprocessing import build_datasets
from src.model.model import build_model, compile_model, enable_fine_tuning
from src.model.callbacks import get_callbacks
from src.utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

CONFIG_PATH = "config/config.yaml"
PARAMS_PATH = "params.yaml"


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def compute_class_weights(splits_path: str, class_index: dict) -> dict:
    with open(splits_path, "r") as f:
        splits = json.load(f)

    train_labels = [class_index[cls] for _, cls in splits["train"]]
    classes = np.array(sorted(class_index.values()))
    weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=np.array(train_labels)
    )
    weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
    logger.info(f"Computed class weights (balanced): {weight_dict}")
    return weight_dict


def save_training_plots(history_phase1, history_phase2, out_dir: Path) -> Path:
    """Merge both phase histories and plot loss + accuracy curves."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Merge both histories
    def _extend(base, extra):
        return base + extra

    all_loss     = _extend(history_phase1.history["loss"],         history_phase2.history["loss"])
    all_val_loss = _extend(history_phase1.history["val_loss"],     history_phase2.history["val_loss"])
    all_acc      = _extend(history_phase1.history["accuracy"],     history_phase2.history["accuracy"])
    all_val_acc  = _extend(history_phase1.history["val_accuracy"], history_phase2.history["val_accuracy"])

    warmup_end = len(history_phase1.history["loss"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, train_vals, val_vals, title in [
        (axes[0], all_loss,  all_val_loss, "Loss"),
        (axes[1], all_acc,   all_val_acc,  "Accuracy"),
    ]:
        epochs = range(1, len(train_vals) + 1)
        ax.plot(epochs, train_vals, label=f"train_{title.lower()}")
        ax.plot(epochs, val_vals,   label=f"val_{title.lower()}")
        ax.axvline(x=warmup_end, color="gray", linestyle="--", alpha=0.7,
                   label=f"fine-tune starts (ep {warmup_end + 1})")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()

    fig.suptitle("Training History — Phase 1 (warm-up) + Phase 2 (fine-tune)", fontsize=12)
    fig.tight_layout()
    plot_path = out_dir / "training_history.png"
    fig.savefig(plot_path)
    plt.close(fig)
    logger.info(f"Training history plot saved to {plot_path}")
    return plot_path


def save_combined_history(history_phase1, history_phase2, path: Path):
    """Merge both phase histories into a single JSON for reproducibility."""
    combined = {}
    for key in history_phase1.history:
        combined[key] = history_phase1.history[key] + history_phase2.history.get(key, [])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(combined, f, indent=2)
    logger.info(f"Combined training history saved to {path}")


def setup_mlflow():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    username = os.getenv("DAGSHUB_USERNAME")
    token = os.getenv("DAGSHUB_TOKEN")

    if not tracking_uri or not username or not token:
        logger.warning(
            "MLflow/DagsHub env vars not fully set — falling back to local ./mlruns tracking. "
            "Set MLFLOW_TRACKING_URI, DAGSHUB_USERNAME, DAGSHUB_TOKEN in .env for remote tracking."
        )
        return

    os.environ["MLFLOW_TRACKING_USERNAME"] = username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"MLflow tracking URI set to {tracking_uri}")


def main() -> int:
    start_time = time.time()

    config = load_yaml(CONFIG_PATH)
    params = load_yaml(PARAMS_PATH)

    seed = params["seed"]
    np.random.seed(seed)
    import tensorflow as tf
    tf.random.set_seed(seed)

    setup_mlflow()
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    # Pull training params
    warmup_epochs   = params["train"].get("warmup_epochs", 8)
    warmup_lr       = params["train"].get("warmup_lr", 0.001)
    finetune_epochs = params["train"]["epochs"]
    finetune_lr     = params["train"]["learning_rate"]
    trainable_layers = params["model"]["trainable_layers"]
    model_dir       = config["paths"]["model_dir"]

    try:
        with mlflow.start_run():
            mlflow.log_param("git_commit",        get_git_commit_hash())
            mlflow.log_param("model_name",        params["model"]["name"])
            mlflow.log_param("image_size",        params["model"]["image_size"])
            mlflow.log_param("batch_size",        params["train"]["batch_size"])
            mlflow.log_param("warmup_epochs",     warmup_epochs)
            mlflow.log_param("warmup_lr",         warmup_lr)
            mlflow.log_param("finetune_epochs",   finetune_epochs)
            mlflow.log_param("finetune_lr",       finetune_lr)
            mlflow.log_param("optimizer",         params["train"]["optimizer"])
            mlflow.log_param("dropout",           params["model"]["dropout"])
            mlflow.log_param("augmentation",      params["train"]["augmentation"])
            mlflow.log_param("seed",              seed)
            mlflow.log_param("trainable_layers",  trainable_layers)

            logger.info("Building datasets...")
            train_ds, val_ds, test_ds, class_index = build_datasets(
                splits_path="data/processed/splits.json",
                config_path=CONFIG_PATH,
                params_path=PARAMS_PATH,
            )

            class_weights = compute_class_weights("data/processed/splits.json", class_index)

            # ------------------------------------------------------------------
            # PHASE 1 — Head warm-up (backbone fully frozen)
            # ------------------------------------------------------------------
            logger.info("=" * 60)
            logger.info(f"PHASE 1 — Head warm-up ({warmup_epochs} epochs, LR={warmup_lr})")
            logger.info("=" * 60)

            model = build_model(
                image_size=config["image"]["size"],
                num_classes=len(class_index),
                dropout=params["model"]["dropout"],
            )
            model = compile_model(model, learning_rate=warmup_lr)
            model.summary(print_fn=logger.info)

            # Warm-up callbacks: checkpoint only (no early stopping — we want
            # the full warmup_epochs to run so the head converges properly).
            warmup_callbacks = [
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=str(Path(model_dir) / "best_model.keras"),
                    monitor="val_loss",
                    save_best_only=True,
                    verbose=1,
                ),
            ]

            history_phase1 = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=warmup_epochs,
                class_weight=class_weights,
                callbacks=warmup_callbacks,
            )

            logger.info("Phase 1 complete.")
            val_loss_p1, val_acc_p1 = model.evaluate(val_ds, verbose=0)
            logger.info(f"Phase 1 result — val_loss={val_loss_p1:.4f}, val_acc={val_acc_p1:.4f}")
            mlflow.log_metric("warmup_val_loss",     val_loss_p1)
            mlflow.log_metric("warmup_val_accuracy", val_acc_p1)

            # Log phase-1 epoch metrics
            for i, (loss, acc, vl, va) in enumerate(zip(
                history_phase1.history["loss"],
                history_phase1.history["accuracy"],
                history_phase1.history["val_loss"],
                history_phase1.history["val_accuracy"],
            )):
                mlflow.log_metrics(
                    {"train_loss": loss, "train_accuracy": acc,
                     "val_loss": vl,   "val_accuracy": va},
                    step=i,
                )

            # ------------------------------------------------------------------
            # PHASE 2 — Fine-tuning (last N backbone layers unfrozen)
            # ------------------------------------------------------------------
            logger.info("=" * 60)
            logger.info(f"PHASE 2 — Fine-tuning ({finetune_epochs} epochs, LR={finetune_lr})")
            logger.info(f"         Unfreezing last {trainable_layers} backbone layers")
            logger.info("=" * 60)

            model = enable_fine_tuning(model, trainable_layers=trainable_layers)
            model = compile_model(model, learning_rate=finetune_lr)

            finetune_callbacks = get_callbacks(model_dir, patience=7)

            history_phase2 = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=finetune_epochs,
                class_weight=class_weights,
                callbacks=finetune_callbacks,
            )

            logger.info("Phase 2 complete.")
            val_loss_p2, val_acc_p2 = model.evaluate(val_ds, verbose=0)
            logger.info(f"Phase 2 result — val_loss={val_loss_p2:.4f}, val_acc={val_acc_p2:.4f}")
            mlflow.log_metric("final_val_loss",     val_loss_p2)
            mlflow.log_metric("final_val_accuracy", val_acc_p2)

            # Log phase-2 epoch metrics (offset step by warmup_epochs)
            for i, (loss, acc, vl, va) in enumerate(zip(
                history_phase2.history["loss"],
                history_phase2.history["accuracy"],
                history_phase2.history["val_loss"],
                history_phase2.history["val_accuracy"],
            )):
                mlflow.log_metrics(
                    {"train_loss": loss, "train_accuracy": acc,
                     "val_loss": vl,   "val_accuracy": va},
                    step=warmup_epochs + i,
                )

            # ------------------------------------------------------------------
            # Save artifacts
            # ------------------------------------------------------------------
            final_path = Path(model_dir) / "final_model.keras"
            model.save(final_path)
            logger.info(f"Final model saved to {final_path}")

            history_path = (
                Path(config["paths"]["reports_dir"]) / "metrics" / "training_history.json"
            )
            save_combined_history(history_phase1, history_phase2, history_path)

            plot_path = save_training_plots(
                history_phase1,
                history_phase2,
                Path(config["paths"]["reports_dir"]) / "figures",
            )

            # MLflow artifacts
            mlflow.log_artifact(str(plot_path))
            mlflow.log_artifact(str(history_path))
            mlflow.log_artifact("reports/metrics/dataset_summary.json")
            mlflow.log_artifact("data/processed/splits.json")

            logger.info("Logging model to MLflow...")
            mlflow.keras.log_model(model, artifact_path="model")

            elapsed = time.time() - start_time
            mlflow.log_metric("training_time_seconds", elapsed)
            logger.info(f"Total training time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

        return 0

    except Exception:
        logger.exception("Training failed with an exception.")
        return 1


if __name__ == "__main__":
    sys.exit(main())