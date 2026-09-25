import random
import numpy as np
import tensorflow as tf
from pathlib import Path
from data import load_data, make_dataset
from models import build_mobilenet_v2

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
EPOCHS = 30
SEED = 67

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

(x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data()

train_ds = make_dataset(x_train, y_train, train=True)
val_ds = make_dataset(x_val, y_val)

model = build_mobilenet_v2()
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=7,
        restore_best_weights=True,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=3,
        min_lr=1e-6,
    ),
]

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

MODELS_DIR.mkdir(exist_ok=True)
model.save(str(MODELS_DIR / "mobilenet_v2.keras"))
