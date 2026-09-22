# Bit-Width Analysis

This analysis uses signed INT8 inputs and weights and a worst-case bound in which every product has the maximum absolute INT8 value and all products have the same sign.

## INT8 product

Signed INT8 values have the range:

```text
-128 ... 127
```

The largest absolute product is:

```text
127 × 127 = 16,129
```

The product therefore needs 15 signed bits because:

```text
signed 14-bit maximum = 8,191
signed 15-bit maximum = 16,383
```

## Accumulator formula

For an accumulation length of `L` products:

```text
maximum absolute accumulator = L × 16,129
```

The minimum signed width is the smallest `n` satisfying:

```text
2^(n-1) - 1 ≥ L × 16,129
```

## Relevant accumulation lengths

| Operation | Maximum accumulation length | Maximum absolute sum | Minimum signed width |
|---|---:|---:|---:|
| Initial 3×3 convolution | 3×3×3 = 27 | 435,483 | 20 bits |
| Depthwise 3×3 convolution | 3×3 = 9 | 145,161 | 19 bits |
| V1 pointwise convolution | 512 input channels | 8,258,048 | 24 bits |
| V2 expanded pointwise/depthwise projection | 480 input channels | 7,741,920 | 24 bits |
| V1 final classifier | 512 features | 8,258,048 | 24 bits |
| V2 final classifier | 640 features | 10,322,560 | 25 bits |

The largest analyzed accumulation is the V2 final classifier:

```text
640 × 16,129 = 10,322,560
```

A signed 25-bit value can represent up to:

```text
2^24 - 1 = 16,777,215
```

Therefore, the theoretical worst-case accumulator for the analyzed layers fits within 25 bits and is safely represented by the INT32 tensors visible in the TFLite graphs.

## Bias

The quantized TFLite graphs use INT32 bias tensors. Bias values are added to accumulated convolution or fully connected results, so INT32 provides a common safe representation for the inference graph.

## Conclusion

For the analyzed operations:

```text
Product width:     at most 15 signed bits
Accumulator width: at most 25 signed bits
Bias width:        INT32
Required ceiling:  below 32 bits
```

This is a theoretical worst-case bound based on the INT8 range and the largest reduction lengths in the corrected alpha=0.5 architectures. Actual trained values are normally much smaller than this bound.
