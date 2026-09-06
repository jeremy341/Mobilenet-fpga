from pathlib import Path
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"


def representative_dataset():
    (x_train, _), _ = tf.keras.datasets.cifar10.load_data()
    for i in range(200):
        img = tf.image.resize(x_train[i].astype("float32"), (128, 128))
        img = img / 255.0
        img = tf.expand_dims(img, 0)
        yield [img]


def quantize_model(model_path, tflite_path, calib_data):
    model = tf.keras.models.load_model(model_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = calib_data
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    tflite_path.parent.mkdir(exist_ok=True)
    tflite_path.write_bytes(tflite_model)


if __name__ == "__main__":
    models = [
        ("mobilenet_v1.keras", "mobilenet_v1_quantized.tflite"),
        ("mobilenet_v2.keras", "mobilenet_v2_quantized.tflite"),
    ]

    for keras_name, tflite_name in models:
        model_path = MODELS_DIR / keras_name
        tflite_path = RESULTS_DIR / tflite_name
        print(f"Quantizing {keras_name}...")
        quantize_model(model_path, tflite_path, representative_dataset)
        print(f"  Saved to {tflite_path}")