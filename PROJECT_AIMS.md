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
