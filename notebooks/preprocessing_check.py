# paste into notebooks/preprocessing_check.py, or a new scratch file
from src.model.model import build_model, compile_model

model = build_model(image_size=224, num_classes=4, dropout=0.3, fine_tune_enabled=False)
model = compile_model(model, learning_rate=1e-4)
model.summary()

import numpy as np
dummy = np.random.rand(2, 224, 224, 3).astype("float32") * 255
preds = model.predict(dummy)
print("Output shape:", preds.shape)
print("Sums to 1:", np.allclose(preds.sum(axis=1), 1.0))