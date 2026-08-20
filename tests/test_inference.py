"""
tests/test_inference.py
-----------------------
Unit tests for preprocessing and inference engine.
"""

import io
import numpy as np
import pytest
from PIL import Image

from src.inference.predict import preprocess_image, predict_image, CLASS_NAMES


@pytest.fixture
def sample_pil_image():
    """Create a temporary synthetic RGB PIL image."""
    arr = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture
def sample_image_bytes(sample_pil_image):
    """Convert PIL image to JPEG bytes."""
    buf = io.BytesIO()
    sample_pil_image.save(buf, format="JPEG")
    return buf.getvalue()


def test_preprocess_image_pil(sample_pil_image):
    """Test preprocessing with PIL.Image input."""
    tensor = preprocess_image(sample_pil_image)
    assert tensor.shape == (1, 224, 224, 3)
    assert tensor.dtype == np.float32


def test_preprocess_image_bytes(sample_image_bytes):
    """Test preprocessing with raw image bytes."""
    tensor = preprocess_image(sample_image_bytes)
    assert tensor.shape == (1, 224, 224, 3)
    assert tensor.dtype == np.float32


def test_predict_image_structure(sample_image_bytes):
    """Test predict_image returns valid schema and probability distribution."""
    result = predict_image(image_bytes=sample_image_bytes)
    
    assert "prediction" in result
    assert result["prediction"] in CLASS_NAMES
    assert "confidence" in result
    assert 0.0 <= result["confidence"] <= 1.0
    
    assert "probabilities" in result
    assert len(result["probabilities"]) == 4
    for class_name in CLASS_NAMES:
        assert class_name in result["probabilities"]
        assert 0.0 <= result["probabilities"][class_name] <= 1.0
    
    # Softmax probabilities should sum to approximately 1.0
    prob_sum = sum(result["probabilities"].values())
    assert pytest.approx(prob_sum, rel=1e-3) == 1.0


def test_invalid_image_bytes_raises_error():
    """Corrupted/invalid bytes should raise ValueError."""
    with pytest.raises(ValueError):
        predict_image(image_bytes=b"not_a_real_image_data")
