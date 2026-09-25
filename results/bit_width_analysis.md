# Bit-width analysis

This estimate follows the LiteRT INT8 convolution and fully connected specification. Activations use signed INT8 `[-128,127]`; symmetric weights use INT8 `[-127,127]` with zero-point 0. [LiteRT 8-bit quantization specification](https://developers.google.com/edge/litert/conversion/tensorflow/quantization/quantization_spec)

## Product width

The largest absolute activation-weight product is:

```text
max(|-128 × 127|, |127 × 127|) = 16,256
```

A signed 15-bit integer covers `[-16,384,16,383]`, so every individual product fits in 15 signed bits.

## Accumulator estimate

For a reduction of `L` products, a conservative same-sign bound is:

```text
maximum absolute sum = L × 16,256
```

The minimum signed width is the smallest `n` for which the signed range `[-2^(n-1), 2^(n-1)-1]` contains that bound.

| Operation | Reduction length | Absolute sum bound | Minimum signed width |
|---|---:|---:|---:|
| Initial 3×3 convolution | `3×3×3 = 27` | 438,912 | 20 bits |
| Depthwise 3×3 convolution | `3×3 = 9` | 146,304 | 19 bits |
| V1 pointwise convolution | 512 input channels | 8,323,072 | 24 bits |
| V2 projection from expanded channels | 480 input channels | 7,802,880 | 24 bits |
| V1 final classifier | 512 features | 8,323,072 | 24 bits |
| V2 final classifier | 640 features | 10,403,840 | 25 bits |

The largest bound is for the V2 classifier. Signed INT25 has maximum `2^24 - 1 = 16,777,215`, which is greater than 10,403,840. Thus the analyzed product sums fit within 25 bits; INT32 accumulators provide additional headroom.

## Bias and exported tensor types

LiteRT specifies INT32 bias for INT8 `CONV_2D`, `DEPTHWISE_CONV_2D`, and `FULLY_CONNECTED` operations. The exported model inspections show INT8 and INT32 tensors only; no tensor in either model exceeds 32 bits. The actual bias values remain model-specific, while their stored datatype is INT32.

## Conclusion

```text
Products:       at most 15 signed bits under the LiteRT weight range
Accumulations:  at most 25 signed bits for the analyzed reductions
Bias tensors:   INT32
Model tensor ceiling: 32 bits
```

This is a conservative range estimate for the corrected `alpha=0.5` models, not a claim that every intermediate takes its maximum value. The product bound relies on the LiteRT symmetric weight range `[-127,127]`; treating both operands as unrestricted signed INT8 would require a different bound.
