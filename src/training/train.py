"""
Full training pipeline: data -> model -> train -> save.
MLflow logging is added in Phase 9 (kept separate so this loop is verifiable
on its own first).

Usage:
    python -m src.training.train
"""
import sys
import json
import time
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")  # no GUI backend needed, just save PNGs
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

from src.preprocessing.preprocessing import build_datasets
from src.model.model import build_model, compile_model
from src.model.callbacks import get_callbacks
from src.utils.logger import get_logger

logger = get_logger(__name__)

CONFIG_PATH = "config/config.yaml"
PARAMS_PATH = "params.yaml"


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


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


def save_training_plots(history, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"], label="train_loss")
    axes[0].plot(history.history["val_loss"], label="val_loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="train_accuracy")
    axes[1].plot(history.history["val_accuracy"], label="val_accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "training_history.png")
    plt.close(fig)
    logger.info(f"Training history plot saved to {out_dir / 'training_history.png'}")


def main() -> int:
    start_time = time.time()

    config = load_yaml(CONFIG_PATH)
    params = load_yaml(PARAMS_PATH)

    seed = params["seed"]
    np.random.seed(seed)
    import tensorflow as tf
    tf.random.set_seed(seed)

    try:
        logger.info("Building datasets...")
        train_ds, val_ds, test_ds, class_index = build_datasets(
            splits_path="data/processed/splits.json",
            config_path=CONFIG_PATH,
            params_path=PARAMS_PATH,
        )

        class_weights = compute_class_weights("data/processed/splits.json", class_index)

        logger.info("Building model...")
        model = build_model(
            image_size=config["image"]["size"],
            num_classes=len(class_index),
            dropout=params["model"]["dropout"],
            fine_tune_enabled=params["model"]["fine_tune_enabled"],
            trainable_layers=params["model"]["trainable_layers"],
        )
        model = compile_model(model, learning_rate=params["train"]["learning_rate"])

        model_dir = config["paths"]["model_dir"]
        callbacks = get_callbacks(model_dir, patience=5)

        logger.info(f"Starting training for up to {params['train']['epochs']} epochs...")
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=params["train"]["epochs"],
            class_weight=class_weights,
            callbacks=callbacks,
        )

        logger.info("Training complete. Evaluating on validation set (final check)...")
        val_loss, val_acc = model.evaluate(val_ds)
        logger.info(f"Final val_loss={val_loss:.4f}, val_accuracy={val_acc:.4f}")

        # save final model explicitly (ModelCheckpoint already saved best_model.keras)
        final_path = Path(model_dir) / "final_model.keras"
        model.save(final_path)
        logger.info(f"Final model saved to {final_path}")

        # save history as json
        history_path = Path(config["paths"]["reports_dir"]) / "metrics" / "training_history.json"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(history_path, "w") as f:
            json.dump(history.history, f, indent=2)
        logger.info(f"Training history saved to {history_path}")

        # save plots
        save_training_plots(history, Path(config["paths"]["reports_dir"]) / "figures")

        elapsed = time.time() - start_time
        logger.info(f"Total training time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

        return 0

    except Exception:
        logger.exception("Training failed with an exception.")
        return 1


if __name__ == "__main__":
    sys.exit(main())