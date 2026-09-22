import unittest

import tensorflow as tf

from src.models import build_mobilenet_v1, build_mobilenet_v2


def first_convolution_filters(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.filters
    raise AssertionError("No Conv2D layer found")


class ModelShapeTests(unittest.TestCase):
    def test_alpha_scales_initial_convolution_for_both_models(self):
        v1 = build_mobilenet_v1(alpha=0.5)
        v2 = build_mobilenet_v2(alpha=0.5)

        self.assertEqual(first_convolution_filters(v1), 16)
        self.assertEqual(first_convolution_filters(v2), 16)

    def test_models_keep_required_input_and_output_shapes(self):
        for builder in (build_mobilenet_v1, build_mobilenet_v2):
            model = builder(alpha=0.5)
            self.assertEqual(model.input_shape, (None, 128, 128, 3))
            self.assertEqual(model.output_shape, (None, 10))


if __name__ == "__main__":
    unittest.main()
