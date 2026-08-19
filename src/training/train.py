"""
Full training pipeline: data -> model -> train -> save -> log to MLflow (DagsHub-hosted).

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
from src.model.model import build_model, compile_model
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


def save_training_plots(history, out_dir: Path) -> Path:
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
    plot_path = out_dir / "training_history.png"
    fig.savefig(plot_path)
    plt.close(fig)
    logger.info(f"Training history plot saved to {plot_path}")
    return plot_path


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

    try:
        with mlflow.start_run():
            mlflow.log_param("git_commit", get_git_commit_hash())
            mlflow.log_param("model_name", params["model"]["name"])
            mlflow.log_param("image_size", params["model"]["image_size"])
            mlflow.log_param("batch_size", params["train"]["batch_size"])
            mlflow.log_param("epochs", params["train"]["epochs"])
            mlflow.log_param("learning_rate", params["train"]["learning_rate"])
            mlflow.log_param("optimizer", params["train"]["optimizer"])
            mlflow.log_param("dropout", params["model"]["dropout"])
            mlflow.log_param("augmentation", params["train"]["augmentation"])
            mlflow.log_param("seed", seed)
            mlflow.log_param("trainable_layers", params["model"]["trainable_layers"])
            mlflow.log_param("fine_tune_enabled", params["model"]["fine_tune_enabled"])

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

            logger.info("Training complete. Evaluating on validation set...")
            val_loss, val_acc = model.evaluate(val_ds)
            logger.info(f"Final val_loss={val_loss:.4f}, val_accuracy={val_acc:.4f}")

            # log per-epoch metrics as MLflow metric history
            for epoch_idx in range(len(history.history["loss"])):
                mlflow.log_metrics(
                    {
                        "train_loss": history.history["loss"][epoch_idx],
                        "train_accuracy": history.history["accuracy"][epoch_idx],
                        "val_loss": history.history["val_loss"][epoch_idx],
                        "val_accuracy": history.history["val_accuracy"][epoch_idx],
                    },
                    step=epoch_idx,
                )

            mlflow.log_metric("final_val_loss", val_loss)
            mlflow.log_metric("final_val_accuracy", val_acc)

            final_path = Path(model_dir) / "final_model.keras"
            model.save(final_path)
            logger.info(f"Final model saved to {final_path}")

            history_path = Path(config["paths"]["reports_dir"]) / "metrics" / "training_history.json"
            history_path.parent.mkdir(parents=True, exist_ok=True)
            with open(history_path, "w") as f:
                json.dump(history.history, f, indent=2)
            logger.info(f"Training history saved to {history_path}")

            plot_path = save_training_plots(history, Path(config["paths"]["reports_dir"]) / "figures")

            # log artifacts to MLflow
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