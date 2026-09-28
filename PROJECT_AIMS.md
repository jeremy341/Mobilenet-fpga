# Project Aims

The goal is to build and compare two efficient image-classification networks: MobileNetV1 and MobileNetV2.

|                        | MobileNetV1 | MobileNetV2 |
| ---------------------- | ----------: | ----------: |
| Input                  | `128×128×3` | `128×128×3` |
| Width multiplier       | `alpha=0.5` | `alpha=0.5` |
| First model            | Float32 | Float32 |
| Afterwards             | INT8 TFLite | INT8 TFLite |
| Input of quantized model | INT8 | INT8 |
| Weights                | INT8 | INT8 |
| Activations            | INT8 | INT8 |
| Measure accuracy       | ✓ | ✓ |
| Measure model size     | ✓ | ✓ |
| Compare parameters     | ✓ | ✓ |
| Inspect with Netron    | ✓ | ✓ |
| Investigate bit widths | ≤32 bit | ≤32 bit |
| Implement on FPGA      | **No** | **No** |

## Required comparison

The final comparison should include:

- Float32 accuracy
- INT8 accuracy
- Accuracy change after quantization
- Float32 model size
- INT8 model size
- Number of parameters
- Tensor and operator data types
- Required bit widths, up to 32 bits

Both models should use the same dataset, input resolution, train/validation/test split, and comparable training settings so that the comparison is meaningful.

The FPGA implementation is explicitly out of scope.

## Learning roadmap

The project should be learned and implemented in the following order. The goal is not to master all of machine learning before writing code; after the simple CNN stage, coding becomes part of the learning process.

### Phase 1 — Neural-network foundations

Learn:

- Weights and biases
- Forward pass
- Loss
- Gradients and backpropagation at a high level
- Optimizers and learning rate
- Batches and epochs
- Accuracy
- Training, validation, and test data
- Overfitting

Resources:

- [3Blue1Brown — But what is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk)
- [3Blue1Brown — Gradient descent](https://www.3blue1brown.com/lessons/gradient-descent/)
- [StatQuest video library](https://statquest.org/video_index.html)

The minimum understanding required is:

```text
Forward pass produces a prediction
Loss measures the error
Backpropagation calculates parameter influence
The optimizer updates the parameters
```

Backpropagation does not need to be implemented from scratch.

### Phase 2 — Tensors, images, and NumPy

Understand image and batch shapes:

```text
(128, 128, 3)       one RGB image
(32, 128, 128, 3)   batch of 32 RGB images
```

Practice `shape`, `dtype`, `min`, `max`, `mean`, `astype`, indexing, `argmax`, and `expand_dims`.

### Phase 3 — A normal CNN

Build and train a small CNN before MobileNet:

```text
Image → Conv2D → ReLU → MaxPooling
      → Conv2D → ReLU → GlobalAveragePooling or Flatten
      → Dense → Prediction
```

Learn kernels, filters, feature maps, channels, stride, padding, convolution, pooling, and activation functions.

Resource:

- [TensorFlow CNN tutorial with CIFAR-10](https://www.tensorflow.org/tutorials/images/cnn)

The code should be understood and rewritten rather than copied without understanding.

### Phase 4 — Image-classification workflow

Learn and implement:

```text
Dataset → train/validation/test split → resize → normalize
        → batch → shuffle → prefetch → augmentation
        → train → evaluate → inspect overfitting
```

Resources:

- [TensorFlow image-classification tutorial](https://www.tensorflow.org/tutorials/images/classification)
- [TensorFlow image-loading tutorial](https://www.tensorflow.org/tutorials/load_data/images)
- [TensorFlow data-augmentation tutorial](https://www.tensorflow.org/tutorials/images/data_augmentation)

### Phase 5 — Keras Functional API

Understand inputs, outputs, layer graphs, `keras.Model`, skip connections, and `model.summary()`.

Resource:

- [Keras Functional API guide](https://keras.io/guides/functional_api/)

### Phase 6 — MobileNetV1

Study and understand:

- Standard convolution
- Depthwise convolution
- Pointwise `1×1` convolution
- Depthwise-separable convolution
- Width multiplier `alpha`
- Resolution multiplier

Resources:

- [MobileNets paper](https://arxiv.org/abs/1704.04861)
- [Keras MobileNet documentation](https://keras.io/api/applications/mobilenet/mobilenet_models/)

Implement an educational version using:

```text
conv_block()
depthwise_separable_block()
mobilenet_v1()
```

Verify output shapes and parameter counts against the official implementation.

### Phase 7 — MobileNetV2

Study and understand:

- Inverted residual blocks
- Expansion layers
- Projection layers
- Linear bottlenecks
- ReLU6
- Residual connections

Resources:

- [MobileNetV2 paper](https://arxiv.org/abs/1801.04381)
- [Keras MobileNetV2 documentation](https://keras.io/api/applications/mobilenet/mobilenet_models/)

Implement an educational version using an inverted-residual block and verify its shapes and parameter count.

### Phase 8 — Dataset selection and training

The selected and permitted project dataset is CIFAR-10. Candidate alternatives considered during planning were:

- [CIFAR-10](https://www.tensorflow.org/datasets/catalog/cifar10) for initial practice
- [Oxford-IIIT Pet](https://www.tensorflow.org/datasets/catalog/oxford_iiit_pet) for a manageable animal-classification project
- [Food-101](https://www.tensorflow.org/datasets/catalog/food101) for a larger food-classification project
- [TensorFlow Datasets catalog](https://www.tensorflow.org/datasets/catalog/overview) for alternatives

The same CIFAR-10 split and preprocessing are used for V1 and V2.

Train both models with:

```text
Input: 128×128×3
alpha: 0.5
Same preprocessing
Same train/validation/test split
Comparable training settings
```

### Phase 9 — Float32 baseline

Save and measure both Float32 models before quantization:

```text
models/mobilenet_v1.keras
models/mobilenet_v2.keras
models/mobilenet_v1_float.tflite
models/mobilenet_v2_float.tflite
```

Record accuracy, parameter count, model size, and training settings.

### Phase 10 — INT8 quantization

Learn:

- FP32 versus INT8
- Signed INT8 range (`-128` to `127`)
- Scale
- Zero-point
- Per-tensor and per-channel quantization
- INT8 weights and activations
- INT32 bias and accumulator arithmetic

The basic relationship is:

```text
real_value = (quantized_value - zero_point) × scale
```

Resources:

- [LiteRT post-training integer quantization](https://ai.google.dev/edge/litert/models/post_training_integer_quant)
- [LiteRT 8-bit quantization specification](https://ai.google.dev/edge/litert/performance/quantization_spec)
- [Keras quantization guides](https://keras.io/guides/)

Use a representative subset of real training images for calibration and force integer-only operators where required.

### Phase 11 — TFLite/LiteRT verification

Create:

```text
models/mobilenet_v1_quantized.tflite
models/mobilenet_v2_quantized.tflite
```

Inspect them with the interpreter:

```text
allocate_tensors()
get_input_details()
get_output_details()
get_tensor_details()
set_tensor()
invoke()
get_tensor()
```

Resources:

- [LiteRT Interpreter documentation](https://ai.google.dev/edge/litert/api_docs/python/tf/lite/Interpreter)
- [Netron](https://netron.app/)
- [Netron documentation](https://github.com/lutzroeder/netron)

Verify actual input, output, weight, activation, bias, tensor, and operator dtypes. Do not rely on the filename alone.

### Phase 12 — Evaluation and bit-width analysis

Run the same test set through:

```text
V1 Float32
V1 INT8
V2 Float32
V2 INT8
```

Compare Float32 accuracy, INT8 accuracy, accuracy loss, model sizes, parameter counts, datatypes, and bit widths.

For bit-width analysis, calculate the required width for INT8×INT8 products and sums of products:

```text
product = input × weight
accumulator = Σ(input × weight)
```

Confirm that all required representations remain at or below 32 bits. FPGA implementation remains out of scope.

## Current project status

Based on the current repository contents:

- [x] V1 and V2 model-builder code exists.
- [x] Input shape and `alpha=0.5` are applied to the initial convolution and later width-controlled stages in the model builders.
- [x] CIFAR-10 loading, resizing, normalization, and train/validation/test splitting exist.
- [x] Float32 training scripts exist.
- [x] Saved Float32 `.keras` models exist locally.
- [x] Quantization code exists.
- [x] Both quantized `.tflite` files exist in the canonical `models/` folder.
- [x] Actual TFLite inspection confirms INT8 model inputs and outputs, with INT8 and INT32 tensors present.
- [x] Accuracy and model-size comparison results are documented.
- [x] Both models were retrained on Kaggle with a maximum of 30 epochs, early stopping, and learning-rate reduction.
- [x] Float32 TFLite models were exported and compared against INT8 TFLite models.
- [x] Final Kaggle results and machine-readable TFLite inspections were generated.
- [x] Corrected V1 and V2 INT8 artifacts were inspected in Netron and documented in `results/netron_inspection.md`.
- [x] A reproducible theoretical bit-width calculation is recorded in `results/bit_width_analysis.md`.
- [x] CIFAR-10 is confirmed as an allowed dataset.
- [x] Corrected Keras, Float32 TFLite, and INT8 TFLite artifacts are organized in the canonical `models/` folder; the four TFLite exports are tracked in GitHub and Keras checkpoints remain local-only.
- [x] Training settings, seed, histories, comparison metrics, Netron findings, bit-width calculations, and datatype inspection are documented.
- [ ] Record Kaggle Python and TensorFlow version metadata if it can be retrieved from the notebook session.

Therefore, the required model, quantization, comparison, Netron, and bit-width work is complete in the local project. The only remaining reproducibility detail is Kaggle runtime-version metadata, if available; FPGA implementation remains out of scope.
