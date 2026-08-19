"""
EfficientNetB0 transfer-learning model for 4-class kidney CT classification.
"""
import tensorflow as tf

from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_model(
    image_size: int,
    num_classes: int,
    dropout: float,
    fine_tune_enabled: bool = False,
    trainable_layers: int = 0,
) -> tf.keras.Model:
    """
    Builds EfficientNetB0 with a frozen backbone (default) and a custom
    classification head. If fine_tune_enabled, unfreezes the last
    `trainable_layers` layers of the backbone.
    """
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(image_size, image_size, 3),
    )

    base_model.trainable = fine_tune_enabled
    if fine_tune_enabled and trainable_layers > 0:
        # freeze everything except the last N layers
        for layer in base_model.layers[:-trainable_layers]:
            layer.trainable = False
        logger.info(f"Fine-tuning enabled: last {trainable_layers} backbone layers trainable.")
    elif not fine_tune_enabled:
        logger.info("Backbone frozen (fine_tune_enabled=False) — training classification head only.")

    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    x = base_model(inputs, training=fine_tune_enabled)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="kidney_efficientnetb0")
    return model


def compile_model(model: tf.keras.Model, learning_rate: float) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model