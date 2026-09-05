import tensorflow as tf

print("TensorFlow version:", tf.__version__)

print("Loading CIFAR-10...")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

print("\nDataset loaded successfully!")

print("Training images:", x_train.shape)
print("Training labels:", y_train.shape)

print("Test images:", x_test.shape)
print("Test labels:", y_test.shape)

print("Pixel range:", x_train.min(), "to", x_train.max())