#pragma once

#include "pyflame_rt/tensor.hpp"
#include "pyflame_rt/quantization/quant_params.hpp"
#include <vector>

namespace pyflame_rt {
namespace quantization {

// ============================================================================
// Core Quantization Operations
// ============================================================================

/// Quantize float tensor to int8/uint8
/// q = clamp(round(x / scale) + zero_point, qmin, qmax)
Tensor quantize_tensor(const Tensor& input, const QuantParams& params);

/// Dequantize int8/uint8 tensor to float
/// x = (q - zero_point) * scale
Tensor dequantize_tensor(const Tensor& input, const QuantParams& params);

/// Quantize float tensor in-place (modifies the tensor, returns new tensor)
Tensor quantize_tensor_inplace(Tensor& input, const QuantParams& params);

// ============================================================================
// FP16 / BFloat16 Casting
// ============================================================================

/// Cast float32 tensor to float16
Tensor cast_to_fp16(const Tensor& input);

/// Cast float16 tensor to float32
Tensor cast_from_fp16(const Tensor& input);

/// Cast float32 tensor to bfloat16
Tensor cast_to_bfloat16(const Tensor& input);

/// Cast bfloat16 tensor to float32
Tensor cast_from_bfloat16(const Tensor& input);

// ============================================================================
// Dynamic Quantization Helpers
// ============================================================================

/// Compute quantization parameters dynamically from tensor values
/// Uses min/max of tensor values
QuantParams compute_dynamic_params(const Tensor& tensor,
                                    DType target_dtype = DType::Int8,
                                    bool symmetric = true);

/// Compute per-channel quantization parameters dynamically
QuantParams compute_dynamic_params_per_channel(const Tensor& tensor,
                                                int channel_axis,
                                                DType target_dtype = DType::Int8,
                                                bool symmetric = true);

// ============================================================================
// Quantized Arithmetic Operations
// ============================================================================

/// Quantized matrix multiplication
/// Computes: dequant(quantize(A) @ quantize(B))
/// With fused scale computation for efficiency
Tensor quantized_matmul(const Tensor& a, const QuantParams& a_params,
                        const Tensor& b, const QuantParams& b_params,
                        const QuantParams* output_params = nullptr);

/// Quantized element-wise addition
Tensor quantized_add(const Tensor& a, const QuantParams& a_params,
                     const Tensor& b, const QuantParams& b_params,
                     const QuantParams* output_params = nullptr);

/// Quantized convolution
Tensor quantized_conv2d(const Tensor& input, const QuantParams& input_params,
                        const Tensor& weight, const QuantParams& weight_params,
                        const Tensor* bias,
                        const std::vector<int64_t>& pads,
                        const std::vector<int64_t>& strides,
                        const std::vector<int64_t>& dilations,
                        int64_t groups,
                        const QuantParams* output_params = nullptr);

// ============================================================================
// Utility Functions
// ============================================================================

/// Check if a tensor can be quantized
bool can_quantize(const Tensor& tensor);

/// Get recommended quantization parameters based on tensor statistics
QuantParams recommend_params(const Tensor& tensor,
                              DType target_dtype,
                              QuantGranularity granularity,
                              bool symmetric);

/// Compute quantization error (MSE between original and quantized->dequantized)
float compute_quant_error(const Tensor& original,
                          const Tensor& quantized,
                          const QuantParams& params);

} // namespace quantization
} // namespace pyflame_rt
