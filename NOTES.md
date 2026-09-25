# Notes — MobileNetV1 and MobileNetV2

## Dataset and preprocessing

- CIFAR-10: 50,000 training images and 10,000 test images; RGB images are 32×32 with 10 classes.
- Seeded split: 45,000 training / 5,000 validation / 10,000 test (`SEED=67`).
- Resize to 128×128 with `tf.image.resize`, then convert to Float32 and normalize pixels to `[0,1]`.
- Labels are one-hot encoded.
- Batch size: 128.

## Architecture and training

- Both models use `alpha=0.5`, including the initial convolution, so the stem has 16 filters.
- V1 uses 13 depthwise-separable blocks, ReLU, global average pooling, and a 10-class Dense classifier.
- V2 uses 17 inverted-residual blocks across seven stages, ReLU6, linear projections, residual additions when shapes match, global average pooling, and a 10-class classifier.
- Both use categorical cross-entropy, Adam with initial learning rate `1e-3`, and accuracy as a metric.
- Kaggle run: seed 67, maximum 30 epochs, EarlyStopping on validation accuracy (patience 7, restore best weights), ReduceLROnPlateau on validation loss (factor 0.2, patience 3, minimum learning rate `1e-6`).
- Training stopped after 26 epochs for V1 and 23 for V2. Best validation accuracy: V1 76.68% at epoch 19; V2 79.06% at epoch 16.
- The final logged training accuracies approach 100% while validation accuracy plateaus, indicating overfitting. The saved Keras checkpoints are the best-validation checkpoints.

## Model sizes and parameter counts

| Metric | V1 | V2 |
|---|---:|---:|
| Keras parameters | 834,666 | 604,234 |
| Keras model file | 10,297,401 bytes | 7,796,765 bytes |
| Float32 TFLite | 3,290,072 bytes | 2,346,032 bytes |
| INT8 TFLite | 985,568 bytes | 845,224 bytes |

Netron's TFLite graph parameter counts differ from Keras counts because they measure different representations and the converter folds BatchNormalization into inference operations. The Netron counts are V1 818,252 and V2 578,652; use Keras counts consistently for comparing trainable model sizes.

## Quantization and test results

- Full-integer post-training TFLite conversion uses 200 representative CIFAR-10 training images.
- Converter requires `TFLITE_BUILTINS_INT8`; external input and output types are `tf.int8`.
- Both graphs contain only INT8 and INT32 tensors: V1 has 59 INT8 and 29 INT32 tensors; V2 has 119 INT8 and 54 INT32 tensors.
- Both inputs have shape `[1,128,128,3]`; outputs have shape `[1,10]`.
- Input quantization: scale `0.003921568859368563`, zero-point `-128`. Output quantization: scale `0.00390625`, zero-point `-128`.

| Metric | V1 Float32 | V1 INT8 | V2 Float32 | V2 INT8 |
|---|---:|---:|---:|---:|
| Test accuracy | 76.72% | 76.35% | 77.93% | 77.94% |
| Accuracy change | — | -0.37 percentage points | — | +0.01 percentage points |
| Float32-to-INT8 file reduction | — | 70.04% (3.34×) | — | 63.97% (2.78×) |

The V2 INT8 score is one test image higher than its Float32 score; treat this as a tiny measured variation. Full epoch histories, comparison files, and tensor inspection are in `results/`. Netron graph details are in `results/netron_inspection.md`; arithmetic bounds are in `results/bit_width_analysis.md`.
