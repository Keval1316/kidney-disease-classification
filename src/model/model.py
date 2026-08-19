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
    Builds EfficientNetB0 with a frozen backbone and a richer classification
    head. Fine-tuning (unfreezing the last N backbone layers) is applied
    separately in the training script AFTER a warm-up phase so that the new
    random head weights don't destroy pretrained backbone features on epoch 1.

    Head architecture:
        GlobalAveragePooling2D
        → BatchNormalization
        → Dense(256, relu)
        → Dropout(dropout)
        → Dense(num_classes, softmax)
    """
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(image_size, image_size, 3),
    )

    # Always start with a fully frozen backbone; fine-tuning is enabled
    # explicitly in the training script after the warm-up phase completes.
    base_model.trainable = False
    logger.info("Backbone frozen — will be partially unfrozen after warm-up.")

    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    # Pass training=False so BatchNorm layers inside EfficientNet always run in
    # inference mode during warm-up (avoids running mean/variance corruption).
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="kidney_efficientnetb0")
    return model


def enable_fine_tuning(
    model: tf.keras.Model,
    trainable_layers: int,
) -> tf.keras.Model:
    """
    Unfreezes the last `trainable_layers` layers of the EfficientNetB0
    backbone. Call this AFTER the warm-up phase is complete, then recompile
    with a lower learning rate before continuing training.
    """
    # The backbone is always the second layer of our functional model
    base_model = model.layers[1]
    base_model.trainable = True

    # Freeze everything except the last N layers
    for layer in base_model.layers[:-trainable_layers]:
        layer.trainable = False

    trainable_count = sum(1 for l in base_model.layers if l.trainable)
    logger.info(
        f"Fine-tuning enabled: {trainable_count} backbone layers now trainable "
        f"(last {trainable_layers} of {len(base_model.layers)})."
    )
    return model


def compile_model(model: tf.keras.Model, learning_rate: float) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model