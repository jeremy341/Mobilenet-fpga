# Comparison: MobileNetV1 vs MobileNetV2

## Experiment configuration

- Dataset: CIFAR-10
- Input: `128×128×3`
- Width multiplier: `alpha=0.5`
- Training: maximum 30 epochs with early stopping
- Optimizer: Adam with initial learning rate `1e-3`
- Calibration samples: 200 training images
- Hardware: Kaggle NVIDIA Tesla P100

## Accuracy, parameters, and size

| Metric | V1 Float32 | V1 INT8 | V2 Float32 | V2 INT8 |
|---|---:|---:|---:|---:|
| Test accuracy | 76.44% | 76.19% | 78.33% | 78.24% |
| Accuracy change | — | -0.25 percentage points | — | -0.09 percentage points |
| Parameters | 835,882 | 835,882 | 605,066 | 605,066 |
| Keras model size | 10.31 MB | — | 7.81 MB | — |
| Float32 TFLite size | 3.29 MB | — | 2.35 MB | — |
| INT8 TFLite size | — | 988 KB | — | 847 KB |
| Float32-to-INT8 TFLite ratio | — | 3.34× | — | 2.77× |

## TFLite inspection

### MobileNetV1

```text
Input:  INT8, shape [1, 128, 128, 3]
Output: INT8, shape [1, 10]
INT8 tensors:  59
INT32 tensors: 29
Total tensors: 88
```

### MobileNetV2

```text
Input:  INT8, shape [1, 128, 128, 3]
Output: INT8, shape [1, 10]
INT8 tensors:  119
INT32 tensors: 54
Total tensors: 173
```

Both models used the following input and output quantization parameters:

```text
Input scale:       0.003921568859368563
Input zero-point: -128
Output scale:      0.00390625
Output zero-point: -128
```

The complete machine-readable inspection is in `kaggle_results/results/tflite_inspection.json`.

## Interpretation

MobileNetV2 is more accurate in this experiment and uses 27.6% fewer parameters than V1. INT8 quantization causes only a small accuracy reduction for both models. The Float32 TFLite models are substantially larger than their INT8 versions.

The training histories show overfitting: training accuracy continued approaching 100%, while validation accuracy stopped improving around 77.3% for V1 and 79.5% for V2. The reported test results use the best validation checkpoints.

These are the current Kaggle results. They replace the earlier five-epoch preliminary results.
