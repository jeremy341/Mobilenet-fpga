# MobileNet V1 vs. V2

An educational implementation and comparison of MobileNetV1 and MobileNetV2 for image classification.

The project investigates how two lightweight CNN architectures compare before and after full-integer INT8 quantization. The implementation is intended to make the architecture, tensor types, parameter counts, model sizes, and accuracy trade-offs inspectable.

The complete specification and learning roadmap are documented in [PROJECT_AIMS.md](PROJECT_AIMS.md).

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
- Adam optimizer with learning rate `1e-3`
- Five training epochs

The dataset can be changed later after confirming the assignment requirements. Candidate alternatives and selection criteria are listed in [PROJECT_AIMS.md](PROJECT_AIMS.md).

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

## Current results

The currently recorded results are:

| Metric | V1 Float32 | V1 INT8 | V2 Float32 | V2 INT8 |
|---|---:|---:|---:|---:|
| Test accuracy | 60.44% | 60.23% | 63.05% | 63.03% |
| Accuracy change | — | -0.21% | — | -0.02% |
| Parameters | 835,882 | 835,882 | 605,066 | 605,066 |
| Model size | 9.83 MB | 964 KB | 7.44 MB | 826 KB |

Detailed results are available in [results/comparison.md](results/comparison.md). These values should be reproducible by rerunning the training, quantization, and evaluation scripts before being treated as final submission results.

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
│   └── comparison.md    # Recorded comparison results
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

The scripts create or use:

```text
models/mobilenet_v1.keras
models/mobilenet_v2.keras
results/mobilenet_v1_quantized.tflite
results/mobilenet_v2_quantized.tflite
```

## Verification still to complete

The core implementation and initial experiment are present. Before calling the project a final submission, complete these checks:

- Re-run the full experiment from a clean environment.
- Record the exact training configuration and random seeds.
- Inspect both INT8 files in Netron and save the relevant observations.
- Add a reproducible accumulator and required-bit-width calculation.
- Confirm that the current architecture matches the intended MobileNet specifications, especially the placement of the width multiplier.
- Confirm the final dataset with the assignment supervisor.

The project does not include, and does not require, an FPGA implementation.

## Learning resources

The recommended videos, TensorFlow tutorials, Keras documentation, MobileNet papers, LiteRT quantization documentation, and Netron references are collected in [PROJECT_AIMS.md](PROJECT_AIMS.md).
