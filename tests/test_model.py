"""
tests/test_model.py
-------------------
Unit and smoke tests for the EfficientNetB0 transfer learning architecture.
"""

import numpy as np
import pytest
import tensorflow as tf

from src.model.model import build_model, compile_model, enable_fine_tuning


def test_model_construction_and_shapes():
    """Verify input shape, output shape, and backbone freezing."""
    model = build_model(
        image_size=224,
        num_classes=4,
        dropout=0.3,
        fine_tune_enabled=False,
    )
    
    assert model.input_shape == (None, 224, 224, 3)
    assert model.output_shape == (None, 4)
    
    # Backbone layer (index 1) should initially have trainable=False
    backbone = model.layers[1]
    assert backbone.trainable is False


def test_fine_tuning_unfreezing():
    """Verify selective unfreezing of backbone layers."""
    model = build_model(
        image_size=224,
        num_classes=4,
        dropout=0.3,
    )
    model = enable_fine_tuning(model, trainable_layers=20)
    
    backbone = model.layers[1]
    trainable_backbone_layers = [l for l in backbone.layers if l.trainable]
    assert len(trainable_backbone_layers) == 20


def test_tiny_smoke_training_step():
    """
    Run 1 training step on synthetic data to ensure graph and loss computation
    work end-to-end without shape/API breakage.
    """
    model = build_model(image_size=224, num_classes=4, dropout=0.2)
    model = compile_model(model, learning_rate=1e-4)

    # 4 synthetic samples
    x_synthetic = np.random.randn(4, 224, 224, 3).astype(np.float32)
    y_synthetic = np.array([0, 1, 2, 3], dtype=np.int32)

    history = model.fit(x_synthetic, y_synthetic, epochs=1, batch_size=2, verbose=0)
    assert "loss" in history.history
    assert len(history.history["loss"]) == 1
