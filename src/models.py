import tensorflow as tf


def _depthwise_block(x, filters, stride):
    x = tf.keras.layers.DepthwiseConv2D(3, strides=stride, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters, 1, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x


def build_mobilenet_v1(alpha=0.5, input_shape=(128, 128, 3)):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(int(32 * alpha), 3, strides=2, padding="same", use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    blocks = [
        (64, 1, 1),
        (128, 2, 1),
        (128, 1, 1),
        (256, 2, 1),
        (256, 1, 1),
        (512, 2, 1),
        (512, 1, 5),
        (1024, 2, 1),
        (1024, 1, 1),
    ]

    for filters, stride, repeats in blocks:
        filters = int(filters * alpha)
        for _ in range(repeats):
            x = _depthwise_block(x, filters, stride)
            stride = 1

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _inverted_residual_block(x, filters, stride, expand_ratio=6):
    channels = x.shape[-1]
    shortcut = x

    if expand_ratio != 1:
        expanded_channels = channels * expand_ratio
        x = tf.keras.layers.Conv2D(expanded_channels, 1, strides=1, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU(6.0)(x)

    x = tf.keras.layers.DepthwiseConv2D(3, strides=stride, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    x = tf.keras.layers.Conv2D(filters, 1, strides=1, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    if stride == 1 and channels == filters:
        x = tf.keras.layers.Add()([shortcut, x])

    return x


def build_mobilenet_v2(alpha=0.5, input_shape=(128, 128, 3)):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(int(32 * alpha), 3, strides=2, padding="same", use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    blocks = [
        (1, 16, 1, 1),
        (6, 24, 2, 2),
        (6, 32, 3, 2),
        (6, 64, 4, 2),
        (6, 96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    ]

    for t, c, n, s in blocks:
        c = int(c * alpha)
        for i in range(n):
            stride = s if i == 0 else 1
            x = _inverted_residual_block(x, c, stride, expand_ratio=t)

    x = tf.keras.layers.Conv2D(int(1280 * alpha), 1, strides=1, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    print("=== MobileNetV1 ===")
    m1 = build_mobilenet_v1()
    m1.summary()
    print("\n=== MobileNetV2 ===")
    m2 = build_mobilenet_v2()
    m2.summary()
