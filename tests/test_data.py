"""
tests/test_data.py
------------------
Unit tests for data validation, dataset structure, and splits.
"""

from pathlib import Path
import pytest
import numpy as np
import tensorflow as tf
from PIL import Image

from src.preprocessing.preprocessing import build_class_index, _decode_and_resize, _preprocess_effnet
from src.utils.common import load_json


def test_expected_classes():
    """Verify the 4 standard classes are defined and ordered consistently."""
    expected = ["Cyst", "Normal", "Stone", "Tumor"]
    class_idx = build_class_index(expected)
    
    assert sorted(list(class_idx.keys())) == expected
    assert len(class_idx) == 4
    for idx, name in enumerate(expected):
        assert class_idx[name] == idx


def test_splits_file_validity():
    """Check that splits.json exists and contains valid partition keys."""
    splits_path = Path("data/processed/splits.json")
    if splits_path.exists():
        data = load_json(splits_path)
        assert "train" in data
        assert "val" in data
        assert "test" in data
        assert len(data["train"]) > 0
        assert len(data["val"]) > 0
        assert len(data["test"]) > 0
        
        # Verify no intersection between train, val, and test (leakage check)
        train_set = set(p[0] if isinstance(p, list) else p for p in data["train"])
        val_set = set(p[0] if isinstance(p, list) else p for p in data["val"])
        test_set = set(p[0] if isinstance(p, list) else p for p in data["test"])
        assert train_set.isdisjoint(val_set)
        assert train_set.isdisjoint(test_set)
        assert val_set.isdisjoint(test_set)
    else:
        pytest.skip("data/processed/splits.json not generated yet; skipping split file test.")


def test_synthetic_image_decoding_and_resizing(tmp_path):
    """Test reading and decoding a generated synthetic image with tensorflow ops."""
    img_file = tmp_path / "test_kidney.jpg"
    img_data = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    Image.fromarray(img_data).save(img_file)

    tensor_img, label = _decode_and_resize(str(img_file), tf.constant(2), image_size=224)
    assert tensor_img.shape == (224, 224, 3)
    assert int(label.numpy()) == 2

    preprocessed_img, label = _preprocess_effnet(tensor_img, label)
    assert preprocessed_img.shape == (224, 224, 3)
