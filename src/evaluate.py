from pathlib import Path
import numpy as np
import tensorflow as tf
from data import load_data, make_dataset


def evaluate_keras(model_path, test_ds):
    model = tf.keras.models.load_model(model_path)
    loss, acc = model.evaluate(test_ds, verbose=0)
    return loss, acc

def evaluate_tflite(tflite_path, x_test, y_test):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_scale, input_zero = input_details[0]['quantization']
    output_scale, output_zero = output_details[0]['quantization']


    correct = 0
    for i in range(len(x_test)):
        # Preprocces one image
        img = tf.image.resize(x_test[i].astype("float32"), (128, 128))
        img = img / 255.0
        img = tf.expand_dims(img, 0).numpy()

        # Quantize input to INT8
        q_img   = (img / input_scale + input_zero).astype(np.int8)

        interpreter.set_tensor(input_details[0]['index'], q_img)
        interpreter.invoke()

        # Dequantize output back to float32
        q_output = interpreter.get_tensor(output_details[0]['index'])
        output = (q_output.astype(np.float32) - output_zero) * output_scale

        pred = np.argmax(output[0])
        true = y_test[i].item()
        if pred == true:
            correct += 1

    return correct / len(x_test)

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    MODELS_DIR = ROOT / "models"
    RESULTS_DIR = ROOT / "results"

    (_, _), (_, _), (x_test, y_test) = load_data()

    test_ds = make_dataset(x_test, y_test)

    for name, keras_file, tflite_file in [
        ("V1", "mobilenet_v1.keras", "mobilenet_v1_quantized.tflite"),
        ("V2", "mobilenet_v2.keras", "mobilenet_v2_quantized.tflite"),
    ]:
        loss, acc = evaluate_keras(MODELS_DIR / keras_file, test_ds)
        print(f"{name} Float32: loss={loss:.4f}, acc={acc:.4f}")

        tflite_acc = evaluate_tflite(RESULTS_DIR / tflite_file, x_test, y_test)
        print(f"{name} INT8:   acc={tflite_acc:.4f}")
