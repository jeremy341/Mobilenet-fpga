# MobileNet V1 vs. V2

An educational implementation and comparison of MobileNetV1 and MobileNetV2 for image classification.

The project investigates how two lightweight CNN architectures compare before and after full-integer INT8 quantization. The implementation is intended to make the architecture, tensor types, parameter counts, model sizes, and accuracy trade-offs inspectable.

The complete specification and learning roadmap are documented in [PROJECT_AIMS.md](PROJECT_AIMS.md).

For the German presentation explanation, see [PRESENTATION_GUIDE_DE.md](PRESENTATION_GUIDE_DE.md).

## Project requirements

| Requirement | MobileNetV1 | MobileNetV2 |
|---|---:|---:|
| Input | `128×128×3` | `128×128×3` |
| Width multiplier | `alpha=0.5` | `alpha=0.5` |
| Training model | Float32 | Float32 |
| Quantized model | Full INT8 TFLite | Full INT8 TFLite |
| Input, weights, activations | INT8 | INT8 |
| Bias and accumulator arithmetic | INT32 | INT32 |
| FPGA implementation | Out of scope | Out of scope |

Both models should use the same dataset, preprocessing, data split, and comparable training settings.

## Current experiment

The current experiment uses CIFAR-10:

- 45,000 training images
- 5,000 validation images
- 10,000 test images
- Original images resized from `32×32×3` to `128×128×3`
- Pixel values normalized to `[0, 1]`
- Adam optimizer with initial learning rate `1e-3`
- Maximum 30 epochs with early stopping and learning-rate reduction
- Kaggle NVIDIA Tesla P100

CIFAR-10 is confirmed as an allowed dataset for this project. Both models use the same split and preprocessing.

## Architecture summary

### MobileNetV1

MobileNetV1 is built from an initial convolution followed by depthwise-separable convolution blocks:

```text
3×3 convolution
→ depthwise 3×3 convolution
→ pointwise 1×1 convolution
→ batch normalization and activation
→ global average pooling
→ classifier
```

### MobileNetV2

MobileNetV2 is built from inverted residual blocks with linear bottlenecks:

```text
1×1 expansion convolution
→ depthwise 3×3 convolution
→ 1×1 linear projection
→ optional residual connection
```

The models are implemented directly in `src/models.py` for learning and inspection rather than using only `keras.applications` as a black box.

At `alpha=0.5`, the initial 32-filter convolution is also scaled to 16 filters in both models.

## Current results

The currently recorded results are:

| Metric | V1 Float32 | V1 INT8 | V2 Float32 | V2 INT8 |
|---|---:|---:|---:|---:|
| Test accuracy | 76.72% | 76.35% | 77.93% | 77.94% |
| Accuracy change | — | -0.37 percentage points | — | +0.01 percentage points |
| Parameters | 834,666 | 834,666 | 604,234 | 604,234 |
| Keras model size | 10.30 MB | — | 7.80 MB | — |
| Float32 TFLite size | 3.29 MB | — | 2.35 MB | — |
| INT8 TFLite size | — | 986 KB | — | 845 KB |
| Float32-to-INT8 TFLite reduction | — | 70.04% (3.34× smaller) | — | 63.97% (2.78× smaller) |

Detailed results are in [results/comparison.md](results/comparison.md), with [Netron findings](results/netron_inspection.md) and the [bit-width calculation](results/bit_width_analysis.md). These corrected results use the alpha-scaled 16-filter initial convolution and best-validation checkpoints. V1 loses 0.37 percentage points after quantization. V2 is 0.01 points higher in INT8, a difference of one image among 10,000 test images. The training histories show overfitting: training accuracy approaches 100% while validation accuracy plateaus.

## Repository structure

```text
.
├── src/
│   ├── data.py          # CIFAR-10 loading and tf.data pipeline
│   ├── models.py        # MobileNetV1 and MobileNetV2 builders
│   ├── train_v1.py      # Float32 V1 training
│   ├── train_v2.py      # Float32 V2 training
│   ├── quantize.py      # Full INT8 TFLite conversion
│   └── evaluate.py      # Float32 and INT8 evaluation
├── results/
│   ├── comparison.md
│   ├── netron_inspection.md
│   ├── bit_width_analysis.md
│   ├── *_float.tflite and *_quantized.tflite
│   ├── *_history.csv
│   ├── comparison.csv / comparison.json
│   └── tflite_inspection.json
├── models/              # Locally generated Float32 model files
├── PROJECT_AIMS.md      # Detailed requirements, roadmap, and status
├── NOTES.md             # Experiment notes
├── check_dataset.py     # Dataset-loading smoke test
├── pyproject.toml       # Python project configuration
└── uv.lock              # Locked dependency versions
```

Generated model files may be ignored by Git because of their size. They can be recreated using the commands below.

## Setup

Python 3.9 or newer is required. Python 3.12 is recommended for this project.

Using `uv`:

```bash
uv sync
```

Using an already activated virtual environment:

```bash
pip install tensorflow
```

## Running the project

Run the dataset smoke test:

```bash
python check_dataset.py
```

Train MobileNetV1 and save the Float32 model:

```bash
python src/train_v1.py
```

Train MobileNetV2 and save the Float32 model:

```bash
python src/train_v2.py
```

Convert both saved models to full-integer INT8 TFLite models:

```bash
python src/quantize.py
```

Evaluate the Float32 and INT8 models:

```bash
python src/evaluate.py
```

The training and quantization scripts create or use these model paths. Float32 TFLite exports are included from the Kaggle run for the size comparison:

```text
models/mobilenet_v1.keras
models/mobilenet_v2.keras
results/mobilenet_v1_float.tflite
results/mobilenet_v2_float.tflite
results/mobilenet_v1_quantized.tflite
results/mobilenet_v2_quantized.tflite
```

The final corrected Kaggle artifacts have been copied into these canonical paths. Keras model files are ignored by Git because of their size; the TFLite files and reports are included in the repository.

## Verification status

The corrected run has been retrained, evaluated, and inspected. One environment detail is still missing from the downloaded run metadata:

- Record the Kaggle Python and TensorFlow versions if they are available from the notebook session metadata.
- Record the Kaggle Python and TensorFlow versions if the notebook session metadata is available.

When presenting, mention the overfitting visible in the histories and report measured accuracy rather than claiming 90%.

The project does not include, and does not require, an FPGA implementation.

## Learning resources

The recommended videos, TensorFlow tutorials, Keras documentation, MobileNet papers, LiteRT quantization documentation, and Netron references are collected in [PROJECT_AIMS.md](PROJECT_AIMS.md).
