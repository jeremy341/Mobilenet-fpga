from pathlib import Path
from data import load_data, make_dataset
from models import build_mobilenet_v2

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

(x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data()

train_ds = make_dataset(x_train, y_train, train=True)
val_ds = make_dataset(x_val, y_val)

model = build_mobilenet_v2()
model.fit(train_ds, validation_data=val_ds, epochs=5)

MODELS_DIR.mkdir(exist_ok=True)
model.save(str(MODELS_DIR / "mobilenet_v2.keras"))