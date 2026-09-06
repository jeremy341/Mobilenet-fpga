import numpy as np
import tensorflow as tf

IMG_SIZE = 128
SEED = 67
VAL_SIZE = 5000
NUM_CLASSES = 10

def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train, y_test = y_train.squeeze(1), y_test.squeeze(1)

    rng = np.random.default_rng(SEED)
    indices = rng.permutation(x_train.shape[0])
    val_idx, train_idx = indices[:VAL_SIZE], indices[VAL_SIZE:]

    return ((x_train[train_idx], y_train[train_idx]),
            (x_train[val_idx], y_train[val_idx]),
            (x_test, y_test)
    )

def make_dataset(x, y, batch_size=128, train=False):
    def prepare(image, label):
        image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
        image = tf.cast(image, "float32") / 255.0
        label = tf.one_hot(tf.cast(label, tf.int32), NUM_CLASSES)
        label = tf.cast(label, "float32")
        return image, label

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if train:
        ds = ds.shuffle(10000, seed=SEED)
    ds = ds.batch(batch_size)
    ds = ds.map(prepare, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


if __name__ == "__main__":
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data()
    for name, x, y in [("train", x_train, y_train), ("val", x_val, y_val), ("test", x_test, y_test)]:
        batch_x, batch_y = next(iter(make_dataset(x, y, train=(name == "train"))))
        print(f"{name}: images {batch_x.shape}, labels {batch_y.shape}, "
              f"range {tf.reduce_min(batch_x):.2f}-{tf.reduce_max(batch_x):.2f}")