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
- Kaggle run: maximum 30 epochs with EarlyStopping and ReduceLROnPlateau on a Tesla P100
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
| Test accuracy | 76.44% | 76.19% | 78.33% | 78.24% |
| Accuracy change | — | -0.25 percentage points | — | -0.09 percentage points |
| Best validation accuracy | 77.30% | — | 79.50% | — |
| Final training accuracy | ~99.75% | — | ~99.98% | — |
| Parameters | 835,882 | 835,882 | 605,066 | 605,066 |
| Keras model size | 10.31 MB | — | 7.81 MB | — |
| Float32 TFLite size | 3.29 MB | — | 2.35 MB | — |
| INT8 TFLite size | — | 988 KB | — | 847 KB |
| Float32-to-INT8 TFLite ratio | — | 3.34× | — | 2.77× |

## Inspections & Bit Widths
- **Bit-Width Constraints**: All tensors and operations are constrained to ≤32 bits.
- **Float32**: Standard 32-bit floating point representation.
- **INT8 Quantized**:
  - Inputs / Outputs: `int8`
  - Weights & Feature activations: `int8`
  - Accumulators and Biases: `int32`
- Post-quantization accuracy degradation is minimal (-0.25 percentage points for V1, -0.09 percentage points for V2).
- The INT8 TFLite files are approximately 3.34× smaller than Float32 TFLite for V1 and 2.77× smaller for V2.
- The complete Kaggle outputs are stored locally under `kaggle_results/`.
