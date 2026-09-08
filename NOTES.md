# Notes — MobileNet V1 & V2 Comparison

## Dataset
- **CIFAR-10**: 50,000 train + 10,000 test images, 32×32×3, 10 classes
- Train split into 45,000 train / 5,000 validation (shuffled, seed=67)
- Resized to 128×128×3 via `tf.image.resize` (bilinear) in the tf.data pipeline
- Pixel values normalized to [0, 1] (float32)
- Labels one-hot encoded (10 classes)

## Models
- Both use `alpha=0.5` (80-90% parameter reduction vs full MobileNet)
- Both use `categorical_crossentropy` + `Adam(1e-3)` + `accuracy` metric
- Input: 128×128×3

### MobileNetV1 (src/models.py: `build_mobilenet_v1`)
- 1 regular conv → 13 depthwise separable blocks → GlobalAvgPool → Dense(10)
- Total params: 835,882 at alpha=0.5
- Standard ReLU activation, no skip connections (sequential feed-forward stack)

### MobileNetV2 (src/models.py: `build_mobilenet_v2`)
- 1 regular conv → 7 bottleneck stages (17 inverted residual blocks total) → 1x1 conv expansion (640 filters) → GlobalAvgPool → Dense(10)
- Total params: 605,066 at alpha=0.5
- Inverted residual structure with linear bottlenecks (1x1 expansion, 3x3 depthwise conv with ReLU6, 1x1 linear projection without activation)
- Residual skip connections added where input and output channels match and stride is 1

## Quantization
- Full integer INT8 post-training quantization via TFLite converter (`tf.lite.OpsSet.TFLITE_BUILTINS_INT8`)
- Calibration dataset: 200 samples from CIFAR-10 training set
- Input and output data types set to `tf.int8`
- Exported models saved to `results/`

## Comparison Results
| Metric | V1 (Float32) | V1 (INT8) | V2 (Float32) | V2 (INT8) |
|---|---|---|---|---|
| Test accuracy | 60.44% | 60.23% | 63.05% | 63.03% |
| Accuracy change | — | -0.21% | — | -0.02% |
| Train accuracy | 67.79% | — | 74.39% | — |
| Val accuracy | 63.52% | — | 63.92% | — |
| Parameters | 835,882 | 835,882 | 605,066 | 605,066 |
| Model file size | 9.83 MB | 964 KB | 7.44 MB | 826 KB |

## Inspections & Bit Widths
- **Bit-Width Constraints**: All tensors and operations are constrained to ≤32 bits.
- **Float32**: Standard 32-bit floating point representation.
- **INT8 Quantized**:
  - Inputs / Outputs: `int8`
  - Weights & Feature activations: `int8`
  - Accumulators and Biases: `int32`
- Post-quantization accuracy degradation is minimal (-0.21% for V1, -0.02% for V2) while reducing model storage requirements by ~90%.