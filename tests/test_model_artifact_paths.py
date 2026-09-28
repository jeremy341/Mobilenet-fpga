import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ModelArtifactPathTests(unittest.TestCase):
    def test_four_tflite_exports_live_in_models_and_not_results(self):
        expected_models = {
            "mobilenet_v1_float.tflite",
            "mobilenet_v1_quantized.tflite",
            "mobilenet_v2_float.tflite",
            "mobilenet_v2_quantized.tflite",
        }

        model_files = {path.name for path in (ROOT / "models").glob("*.tflite")}
        result_files = {path.name for path in (ROOT / "results").glob("*.tflite")}

        self.assertEqual(model_files, expected_models)
        self.assertEqual(result_files, set())


if __name__ == "__main__":
    unittest.main()
