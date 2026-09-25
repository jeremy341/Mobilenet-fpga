# Netron inspection of the final corrected TFLite models

Inspected in Netron on 2026-09-25. These models were trained with `alpha=0.5`, including the initial 3×3 convolution, which gives the stem 16 filters.

## MobileNetV1

```text
Input:  int8[-1, 128, 128, 3]
Output: int8[-1, 10]
Initial Conv2D filter: int8[16, 3, 3, 3]
Initial Conv2D bias:   int32[16]
Netron graph parameters: 818,252
```

The graph contains an initial Conv2D followed by 13 depthwise-separable blocks. Each block uses DepthwiseConv2D then pointwise 1×1 Conv2D, followed by spatial Mean, FullyConnected, and Softmax. The final classifier uses INT8 weights `int8[10,512]` and an INT32 bias `int32[10]`.

## MobileNetV2

```text
Input:  int8[-1, 128, 128, 3]
Output: int8[-1, 10]
Initial Conv2D filter: int8[16, 3, 3, 3]
Initial Conv2D bias:   int32[16]
Residual Add nodes: 10
Netron graph parameters: 578,652
```

The graph shows the V2 pattern: 1×1 expansion Conv2D, DepthwiseConv2D, linear 1×1 projection Conv2D, and Add where input and output shapes match. Intermediate expanded representations reach 480 channels; the final feature convolution has 640 channels. The classifier uses INT8 weights `int8[10,640]` and INT32 bias `int32[10]`.

## Quantization and tensor inventory

Both models show these graph-property expressions:

```text
Input:  real = 0.003921568859368563 × (q + 128)
Output: real = 0.00390625 × (q + 128)
```

Thus the input scale is `1/255`, input zero-point `-128`, output scale `1/256`, and output zero-point `-128`.

The TFLite interpreter inventory agrees with Netron:

| Model | INT8 tensors | INT32 tensors | Total |
|---|---:|---:|---:|
| V1 | 59 | 29 | 88 |
| V2 | 119 | 54 | 173 |

No other tensor dtypes appeared in these two graphs.

## Keras and Netron parameter counts

Keras reports 834,666 parameters for V1 and 604,234 for V2. Netron reports 818,252 and 578,652 parameters in the TFLite graphs. These numbers refer to different representations: conversion folds BatchNormalization and other constants into inference operations. The model comparison therefore uses Keras counts consistently; the Netron counts are recorded as graph metrics.
