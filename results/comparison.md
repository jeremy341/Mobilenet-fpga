# Comparison: MobileNetV1 vs MobileNetV2

Trained on CIFAR-10 (resized 128×128×3), α=0.5, 5 epochs, Adam(1e-3).

| Metric | V1 (Float32) | V1 (INT8) | V2 (Float32) | V2 (INT8) |
|---|---|---|---|---|
| Test accuracy | 60.44% | 60.23% | 63.05% | 63.03% |
| Accuracy change | — | -0.21% | — | -0.02% |
| Train accuracy | 67.79% | — | 74.39% | — |
| Val accuracy | 63.52% | — | 63.92% | — |
| Parameters | 835,882 | 835,882 | 605,066 | 605,066 |
| Model file size | 9.83 MB | 964 KB | 7.44 MB | 826 KB |

## Data Types and Bit-Widths
- **Float32 Models (`.keras`)**: 32-bit IEEE float for all weights, biases, and activations.
- **Quantized INT8 Models (`.tflite`)**:
  - Input/Output Tensors: `INT8` (signed 8-bit integer)
  - Weights & Feature Activations: `INT8`
  - Biases / Accumulators: `INT32`
  - All operator and tensor bit widths strictly ≤ 32 bits.