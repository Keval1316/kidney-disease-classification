"""
src/inference/predict.py
-------------------------
Inference module for the Kidney Disease Classifier.

Responsibilities
----------------
1. Load the saved .keras model (cached after first load — no re-loading on
   every request).
2. Preprocess a raw image (file path, bytes, or PIL Image) using exactly
   the same pipeline as training:
     resize -> float32 -> efficientnet.preprocess_input
3. Run model.predict()
4. Map predicted index -> class name
5. Return a structured dict:
   {
       "prediction":   "Tumor",
       "confidence":   0.9421,
       "probabilities": {"Cyst": 0.02, "Normal": 0.01, "Stone": 0.02, "Tumor": 0.94}
   }

Usage
-----
    # From a file path
    from src.inference.predict import predict_image
    result = predict_image("path/to/image.jpg")

    # From raw bytes (FastAPI UploadFile)
    result = predict_image(image_bytes=file_bytes)

    # Pre-load model explicitly (optional — happens automatically on first call)
    from src.inference.predict import load_model_cached
    load_model_cached()

Class index
-----------
The class-to-index mapping is fixed alphabetically at training time:
    Cyst -> 0,  Normal -> 1,  Stone -> 2,  Tumor -> 3
This must match the mapping used in preprocessing.py::build_class_index().
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Union

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants — must match preprocessing.py and config.yaml
# ---------------------------------------------------------------------------
IMAGE_SIZE: int = 224  # pixels (H x W)

# Alphabetically sorted — same as build_class_index() in preprocessing.py
CLASS_NAMES: list[str] = ["Cyst", "Normal", "Stone", "Tumor"]

# Default model path (resolved relative to repo root at import time)
_DEFAULT_MODEL_PATH = Path("artifacts/model/best_model.keras")

# Module-level model cache so we load weights only once per process
_MODEL_CACHE: dict = {}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_cached(model_path: str | Path | None = None):
    """
    Load and cache the Keras model.

    The model is loaded only once per Python process.  Subsequent calls
    return the cached instance immediately.

    Parameters
    ----------
    model_path : str | Path | None
        Path to a .keras model file.  Defaults to ``artifacts/model/best_model.keras``.
        Can also be overridden via the ``MODEL_PATH`` environment variable.

    Returns
    -------
    keras.Model
    """
    # Resolve path: explicit arg > env var > default
    if model_path is None:
        model_path = Path(os.getenv("MODEL_PATH", str(_DEFAULT_MODEL_PATH)))
    else:
        model_path = Path(model_path)

    cache_key = str(model_path.resolve())

    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Run the training pipeline first (dvc repro) or set MODEL_PATH."
        )

    logger.info("Loading model from %s ...", model_path)
    import tensorflow as tf  # deferred import — keeps startup fast when TF is absent
    model = tf.keras.models.load_model(str(model_path))
    logger.info("Model loaded. Input shape: %s", model.input_shape)

    _MODEL_CACHE[cache_key] = model
    return model


def unload_model() -> None:
    """Clear the model cache (useful in tests to force a fresh load)."""
    _MODEL_CACHE.clear()
    logger.debug("Model cache cleared.")


# ---------------------------------------------------------------------------
# Preprocessing (mirrors the training pipeline exactly)
# ---------------------------------------------------------------------------

def preprocess_image(image_input: Union[str, Path, bytes, "PIL.Image.Image"]) -> "np.ndarray":
    """
    Preprocess a single image into a (1, 224, 224, 3) float32 batch tensor.

    The preprocessing steps must match training exactly:
        1. Decode to RGB
        2. Resize to IMAGE_SIZE x IMAGE_SIZE
        3. Cast to float32 (keeps 0–255 range)
        4. Apply efficientnet.preprocess_input (centers around 0)

    Parameters
    ----------
    image_input : str | Path | bytes | PIL.Image.Image
        - str/Path  → file path on disk
        - bytes     → raw image bytes (e.g. from UploadFile.read())
        - PIL.Image → already-opened image

    Returns
    -------
    np.ndarray, shape (1, 224, 224, 3), dtype float32
    """
    import tensorflow as tf
    from tensorflow.keras.applications.efficientnet import preprocess_input

    # --- 1. Decode to a uint8 numpy array ----------------------------------
    if isinstance(image_input, (str, Path)):
        raw = tf.io.read_file(str(image_input))
        img = tf.io.decode_image(raw, channels=3, expand_animations=False)
        img = img.numpy()  # uint8 HxWx3

    elif isinstance(image_input, bytes):
        raw = tf.io.decode_image(
            tf.constant(image_input), channels=3, expand_animations=False
        )
        img = raw.numpy()  # uint8 HxWx3

    else:
        # PIL.Image
        import PIL.Image as PILImage
        pil = image_input.convert("RGB")
        img = np.array(pil, dtype=np.uint8)  # HxWx3

    # --- 2. Resize ---------------------------------------------------------
    img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE]).numpy()  # float32

    # --- 3. Expand batch dimension -----------------------------------------
    img = np.expand_dims(img, axis=0)  # (1, H, W, 3)

    # --- 4. EfficientNetB0 preprocessing (0-255 float input expected) ------
    img = preprocess_input(img)  # in-place normalisation

    return img.astype(np.float32)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_image(
    image_input: Union[str, Path, bytes, "PIL.Image.Image", None] = None,
    *,
    image_path: str | Path | None = None,
    image_bytes: bytes | None = None,
    model_path: str | Path | None = None,
    class_names: list[str] | None = None,
) -> dict:
    """
    Run inference on a single image.

    Accepts the image in multiple forms for convenience:

    * ``predict_image("path/to/img.jpg")``          — positional path
    * ``predict_image(image_bytes=b"...")``          — raw bytes
    * ``predict_image(image_path="path/to/img")``   — keyword path

    Parameters
    ----------
    image_input : str | Path | bytes | PIL.Image | None
        Image source (positional convenience).
    image_path : str | Path | None
        Keyword-only file path (takes precedence over image_input if set).
    image_bytes : bytes | None
        Keyword-only raw bytes (takes precedence over image_input if set).
    model_path : str | Path | None
        Override model path (uses cached default otherwise).
    class_names : list[str] | None
        Override class name list (uses CLASS_NAMES constant by default).

    Returns
    -------
    dict
        {
            "prediction":    "Tumor",
            "confidence":    0.9421,
            "probabilities": {
                "Cyst":   0.0247,
                "Normal": 0.0112,
                "Stone":  0.0220,
                "Tumor":  0.9421,
            }
        }

    Raises
    ------
    ValueError
        If no image source is provided or the image cannot be decoded.
    FileNotFoundError
        If model file does not exist.
    """
    names = class_names or CLASS_NAMES

    # --- Resolve image source ----------------------------------------------
    if image_path is not None:
        source = image_path
    elif image_bytes is not None:
        source = image_bytes
    elif image_input is not None:
        source = image_input
    else:
        raise ValueError(
            "Provide an image via image_input, image_path, or image_bytes."
        )

    # --- Load model (cached) -----------------------------------------------
    model = load_model_cached(model_path)

    # --- Preprocess --------------------------------------------------------
    try:
        img_batch = preprocess_image(source)
    except Exception as exc:
        raise ValueError(f"Failed to preprocess image: {exc}") from exc

    # --- Predict -----------------------------------------------------------
    logger.debug("Running model.predict() on batch shape %s", img_batch.shape)
    probs = model.predict(img_batch, verbose=0)[0]  # shape: (num_classes,)

    predicted_idx: int = int(np.argmax(probs))
    confidence: float = float(probs[predicted_idx])
    prediction: str = names[predicted_idx]

    probabilities = {name: float(probs[i]) for i, name in enumerate(names)}

    result = {
        "prediction": prediction,
        "confidence": round(confidence, 6),
        "probabilities": {k: round(v, 6) for k, v in probabilities.items()},
    }

    logger.info(
        "Prediction: %s  (confidence=%.2f%%)", prediction, confidence * 100
    )

    return result


# ---------------------------------------------------------------------------
# CLI entry point — quick test from terminal
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python -m src.inference.predict <image_path> [model_path]")
        sys.exit(1)

    img_path = sys.argv[1]
    mdl_path = sys.argv[2] if len(sys.argv) > 2 else None

    result = predict_image(img_path, model_path=mdl_path)
    print(json.dumps(result, indent=2))
