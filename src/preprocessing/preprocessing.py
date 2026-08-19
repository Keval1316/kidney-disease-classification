"""
tf.data pipeline: loads images from splits.json, resizes, applies
EfficientNetB0 preprocessing, batches, and augments (train only).

Usage (as a module, not standalone):
    from src.preprocessing.preprocessing import build_datasets
"""
import json
from pathlib import Path

import tensorflow as tf
import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)

CONFIG_PATH = Path("config/config.yaml")


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def load_splits(splits_path: Path) -> dict:
    with open(splits_path, "r") as f:
        return json.load(f)


def build_class_index(expected_classes: list[str]) -> dict[str, int]:
    # Fixed alphabetical order so index mapping is stable across runs
    ordered = sorted(expected_classes)
    return {c: i for i, c in enumerate(ordered)}


def _decode_and_resize(filepath: tf.Tensor, label: tf.Tensor, image_size: int):
    image = tf.io.read_file(filepath)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, [image_size, image_size])
    return image, label


def _preprocess_effnet(image: tf.Tensor, label: tf.Tensor):
    # EfficientNet's own preprocess_input (expects 0-255 range float input)
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    return image, label

_augmentation_layer = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(factor=0.03),   # ~±10 degrees
    tf.keras.layers.RandomZoom(height_factor=0.1, width_factor=0.1),
    tf.keras.layers.RandomTranslation(height_factor=0.05, width_factor=0.05),
])

def _augment(image: tf.Tensor, label: tf.Tensor):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = _augmentation_layer(image, training=True)
    image = tf.clip_by_value(image, 0.0, 255.0)   # <-- added: keep in valid range before EfficientNet preprocessing
    return image, label


# def _augment(image: tf.Tensor, label: tf.Tensor):
#     # Horizontal flip is anatomically reasonable for axial CT slices
#     # (left/right kidney symmetry) — vertical flip is NOT used, it would
#     # invert superior/inferior orientation in a way that's not realistic.
#     image = tf.image.random_flip_left_right(image)
#     image = tf.image.random_brightness(image, max_delta=0.1)
#     image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
#     # small rotation via random rotation in radians (~10 degrees max)
#     image = tf.image.rot90(image, k=0)  # placeholder no-op kept explicit; see note below
#     return image, label


def _make_dataset(
    pairs: list,
    class_index: dict[str, int],
    image_size: int,
    batch_size: int,
    augment: bool,
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    filepaths = [p[0] for p in pairs]
    labels = [class_index[p[1]] for p in pairs]

    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(filepaths), seed=seed, reshuffle_each_iteration=True)

    ds = ds.map(
        lambda fp, lbl: _decode_and_resize(fp, lbl, image_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if augment:
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.map(_preprocess_effnet, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def build_datasets(
    splits_path: str = "data/processed/splits.json",
    config_path: str = "config/config.yaml",
    params_path: str = "params.yaml",
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, dict]:
    """
    Returns (train_ds, val_ds, test_ds, class_index).
    class_index maps class_name -> integer label, alphabetically fixed.
    """
    config = load_config()
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    image_size = config["image"]["size"]
    batch_size = params["train"]["batch_size"]
    augment = params["train"]["augmentation"]
    seed = params["seed"]

    splits = load_splits(Path(splits_path))
    class_index = build_class_index(config["data"]["expected_classes"])
    logger.info(f"Class index mapping: {class_index}")

    train_pairs = [tuple(p) for p in splits["train"]]
    val_pairs = [tuple(p) for p in splits["val"]]
    test_pairs = [tuple(p) for p in splits["test"]]

    train_ds = _make_dataset(
        train_pairs, class_index, image_size, batch_size,
        augment=augment, shuffle=True, seed=seed,
    )
    val_ds = _make_dataset(
        val_pairs, class_index, image_size, batch_size,
        augment=False, shuffle=False, seed=seed,
    )
    test_ds = _make_dataset(
        test_pairs, class_index, image_size, batch_size,
        augment=False, shuffle=False, seed=seed,
    )

    logger.info(
        f"Datasets built — train batches: {len(train_pairs)//batch_size}, "
        f"val batches: {len(val_pairs)//batch_size}, "
        f"test batches: {len(test_pairs)//batch_size}"
    )

    return train_ds, val_ds, test_ds, class_index