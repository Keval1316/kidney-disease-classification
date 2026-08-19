"""
Training callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint.
"""
from pathlib import Path

import tensorflow as tf


def get_callbacks(model_dir: str, patience: int = 7) -> list:
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            min_delta=0.001,          # ignore improvements smaller than 0.1%
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(2, patience // 3),
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(Path(model_dir) / "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]