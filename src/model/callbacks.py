"""
Training callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint.
"""
from pathlib import Path

import tensorflow as tf


def get_callbacks(model_dir: str, patience: int = 5) -> list:
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(2, patience // 2),
            min_lr=1e-7,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(Path(model_dir) / "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]