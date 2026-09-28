# Final corrected MobileNetV1 vs MobileNetV2 run

## Model artifact locations

All six model artifacts are organized under `models/`. The four versioned TFLite files are `models/mobilenet_v1_float.tflite`, `models/mobilenet_v1_quantized.tflite`, `models/mobilenet_v2_float.tflite`, and `models/mobilenet_v2_quantized.tflite`. The `.keras` training checkpoints are local-only.

## Run configuration

- Dataset: CIFAR-10; 45,000 training, 5,000 validation, 10,000 test images
- Input: `128×128×3`; source images resized from `32×32×3`
- Width multiplier: `alpha=0.5`, including the initial convolution (16 filters)
- Batch size: 128
- Optimizer: Adam, initial learning rate `1e-3`
- Maximum epochs: 30
- Callbacks: EarlyStopping (`val_accuracy`, patience 7, restore best weights) and ReduceLROnPlateau (`val_loss`, factor 0.2, patience 3, minimum LR `1e-6`)
- Representative calibration set: 200 training images
- Training environment: Kaggle GPU notebook

## Results

Accuracy is measured on the same 10,000 CIFAR-10 test images.

| Metric | V1 Float32 | V1 INT8 | V2 Float32 | V2 INT8 |
|---|---:|---:|---:|---:|
| Test accuracy | 76.72% | 76.35% | 77.93% | 77.94% |
| Change from Float32 | — | -0.37 percentage points | — | +0.01 percentage points |
| Keras parameters | 834,666 | 834,666 | 604,234 | 604,234 |
| Keras model size | 10,297,401 bytes (10.30 MB) | — | 7,796,765 bytes (7.80 MB) | — |
| Float32 TFLite size | 3,290,072 bytes (3.29 MB) | — | 2,346,032 bytes (2.35 MB) | — |
| INT8 TFLite size | — | 985,568 bytes (0.99 MB) | — | 845,224 bytes (0.85 MB) |
| Float32-to-INT8 TFLite size reduction | — | 70.04% (3.34× smaller) | — | 63.97% (2.78× smaller) |

MobileNetV2 has 230,432 fewer parameters than V1 (27.61% fewer). Its test accuracy is 1.21 percentage points higher in Float32 and 1.59 points higher in INT8. The INT8 V2 score is 0.01 percentage points higher than its Float32 score, equivalent to one test image; this small difference is quantization variation, not evidence that quantization generally improves accuracy.

## Training history

Early stopping ended the V1 run after 26 epochs and V2 after 23 epochs. Best validation accuracy was 76.68% for V1 at epoch 19 and 79.06% for V2 at epoch 16. The saved Keras checkpoints are selected by validation accuracy.

Training accuracy later approached 100% while validation accuracy stopped improving, indicating overfitting. See `mobilenet_v1_history.csv` and `mobilenet_v2_history.csv` for per-epoch values.

## INT8 TFLite inspection

Both models have INT8 inputs with shape `[1,128,128,3]` and INT8 outputs with shape `[1,10]`.

| Model | INT8 tensors | INT32 tensors | Total tensors |
|---|---:|---:|---:|
| V1 | 59 | 29 | 88 |
| V2 | 119 | 54 | 173 |

Both use these quantization parameters:

```text
Input scale:       0.003921568859368563
Input zero-point: -128
Output scale:      0.00390625
Output zero-point: -128
```

The first INT8 convolution filter in each corrected model has shape `[16,3,3,3]`, confirming that `alpha=0.5` is applied to the stem. The complete machine-readable inspection is in `tflite_inspection.json`; training histories and comparison reports remain under `results/`, while model binaries are under `models/`.

Netron reports 818,252 V1 and 578,652 V2 inference-graph parameters. Keras reports 834,666 and 604,234 training-model parameters; see [`netron_inspection.md`](netron_inspection.md) for the distinction.

The reproducible accumulator-width estimate is documented in [`bit_width_analysis.md`](bit_width_analysis.md).
