#include "pyflame_rt/registry.hpp"
#include "pyflame_rt/tensor.hpp"
#include "pyflame_rt/errors.hpp"
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace pyflame_rt {
namespace ops {

namespace {

// ============================================================================
// Validation Helpers for NN Operators (HIGH-01 fix)
// ============================================================================

/// Validate input count
inline void validate_input_count(const std::vector<const Tensor*>& inputs,
                                  size_t min_count, const char* op_name) {
    if (inputs.size() < min_count) {
        throw std::invalid_argument(
            std::string(op_name) + " requires at least " +
            std::to_string(min_count) + " inputs, got " +
            std::to_string(inputs.size()));
    }
}

/// Validate tensor has expected number of dimensions
inline void validate_ndim(const Tensor& t, size_t expected, const char* name) {
    if (t.ndim() != expected) {
        throw std::invalid_argument(
            std::string(name) + " must have " + std::to_string(expected) +
            " dimensions, got " + std::to_string(t.ndim()));
    }
}

/// Validate tensor has at least N dimensions
inline void validate_min_ndim(const Tensor& t, size_t min_dims, const char* name) {
    if (t.ndim() < min_dims) {
        throw std::invalid_argument(
            std::string(name) + " must have at least " + std::to_string(min_dims) +
            " dimensions, got " + std::to_string(t.ndim()));
    }
}

/// Validate vector attribute has expected size
inline void validate_attr_size(const std::vector<int64_t>& vec, size_t expected,
                                const char* attr_name, const char* op_name) {
    if (!vec.empty() && vec.size() < expected) {
        throw std::invalid_argument(
            std::string(op_name) + ": " + attr_name + " must have at least " +
            std::to_string(expected) + " elements, got " + std::to_string(vec.size()));
    }
}

// ============================================================================
// NN Operators with Validation
// ============================================================================

/// Conv with input validation (HIGH-01 fix)
std::vector<Tensor> cpu_conv(const std::vector<const Tensor*>& inputs,
                             const OpContext& ctx) {
    // Security: validate inputs
    validate_input_count(inputs, 2, "Conv");

    const Tensor& x = *inputs[0];      // [N, C, H, W]
    const Tensor& w = *inputs[1];      // [M, C, kH, kW]
    const Tensor* bias = inputs.size() > 2 ? inputs[2] : nullptr;

    // Security: validate dimensions (HIGH-01)
    validate_ndim(x, 4, "Conv input");
    validate_ndim(w, 4, "Conv weight");

    auto kernel_shape = ctx.node->get_attr<std::vector<int64_t>>("kernel_shape", {});
    auto strides = ctx.node->get_attr<std::vector<int64_t>>("strides", {1, 1});
    auto pads = ctx.node->get_attr<std::vector<int64_t>>("pads", {0, 0, 0, 0});
    auto dilations = ctx.node->get_attr<std::vector<int64_t>>("dilations", {1, 1});
    int64_t group = ctx.node->get_attr<int64_t>("group", 1);

    // Security fix HIGH-08: Validate attribute sizes
    validate_attr_size(strides, 2, "strides", "Conv");
    validate_attr_size(pads, 4, "pads", "Conv");
    validate_attr_size(dilations, 2, "dilations", "Conv");

    int64_t N = x.shape()[0];
    int64_t C = x.shape()[1];
    int64_t H = x.shape()[2];
    int64_t W = x.shape()[3];

    int64_t M = w.shape()[0];

    // Security fix HIGH-08: Validate group parameter
    if (group <= 0) {
        throw std::invalid_argument("Conv: group must be positive, got " + std::to_string(group));
    }
    if (C % group != 0) {
        throw std::invalid_argument(
            "Conv: input channels (" + std::to_string(C) +
            ") must be divisible by group (" + std::to_string(group) + ")");
    }
    if (M % group != 0) {
        throw std::invalid_argument(
            "Conv: output channels (" + std::to_string(M) +
            ") must be divisible by group (" + std::to_string(group) + ")");
    }

    // Security: Validate weight tensor channel dimension matches
    int64_t expected_c_per_group = C / group;
    if (w.shape()[1] != expected_c_per_group) {
        throw std::invalid_argument(
            "Conv: weight tensor channels (" + std::to_string(w.shape()[1]) +
            ") doesn't match expected C/group (" + std::to_string(expected_c_per_group) + ")");
    }

    // HIGH-01 fix: Get kernel dimensions from weight tensor if not provided
    int64_t kH, kW;
    if (kernel_shape.empty()) {
        // Infer from weight tensor shape [M, C/group, kH, kW]
        kH = w.shape()[2];
        kW = w.shape()[3];
    } else {
        if (kernel_shape.size() < 2) {
            throw std::invalid_argument("Conv: kernel_shape must have at least 2 elements");
        }
        kH = kernel_shape[0];
        kW = kernel_shape[1];
        // HIGH-01 fix: Validate kernel_shape matches weight tensor
        if (kH != w.shape()[2] || kW != w.shape()[3]) {
            throw std::invalid_argument(
                "Conv: kernel_shape [" + std::to_string(kH) + ", " + std::to_string(kW) +
                "] doesn't match weight tensor dimensions [" +
                std::to_string(w.shape()[2]) + ", " + std::to_string(w.shape()[3]) + "]");
        }
    }

    int64_t pad_top = pads[0];
    int64_t pad_left = pads[1];
    int64_t stride_h = strides[0];
    int64_t stride_w = strides[1];
    int64_t dil_h = dilations[0];
    int64_t dil_w = dilations[1];

    int64_t outH = (H + 2 * pad_top - dil_h * (kH - 1) - 1) / stride_h + 1;
    int64_t outW = (W + 2 * pad_left - dil_w * (kW - 1) - 1) / stride_w + 1;

    Tensor result({N, M, outH, outW}, x.dtype());
    result.zero();

    const float* x_data = x.data_ptr<float>();
    const float* w_data = w.data_ptr<float>();
    float* out_data = result.data_ptr<float>();

    int64_t C_per_group = C / group;
    int64_t M_per_group = M / group;

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t g = 0; g < group; ++g) {
            for (int64_t m = 0; m < M_per_group; ++m) {
                int64_t m_idx = g * M_per_group + m;
                for (int64_t oh = 0; oh < outH; ++oh) {
                    for (int64_t ow = 0; ow < outW; ++ow) {
                        float sum = 0;
                        for (int64_t c = 0; c < C_per_group; ++c) {
                            int64_t c_idx = g * C_per_group + c;
                            for (int64_t kh = 0; kh < kH; ++kh) {
                                for (int64_t kw = 0; kw < kW; ++kw) {
                                    int64_t ih = oh * stride_h - pad_top + kh * dil_h;
                                    int64_t iw = ow * stride_w - pad_left + kw * dil_w;
                                    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                        int64_t x_idx = ((n * C + c_idx) * H + ih) * W + iw;
                                        int64_t w_idx = ((m_idx * C_per_group + c) * kH + kh) * kW + kw;
                                        sum += x_data[x_idx] * w_data[w_idx];
                                    }
                                }
                            }
                        }
                        int64_t out_idx = ((n * M + m_idx) * outH + oh) * outW + ow;
                        out_data[out_idx] = sum;
                    }
                }
            }
        }
    }

    // Add bias if present
    if (bias) {
        const float* bias_data = bias->data_ptr<float>();
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t m = 0; m < M; ++m) {
                for (int64_t oh = 0; oh < outH; ++oh) {
                    for (int64_t ow = 0; ow < outW; ++ow) {
                        int64_t idx = ((n * M + m) * outH + oh) * outW + ow;
                        out_data[idx] += bias_data[m];
                    }
                }
            }
        }
    }

    return {std::move(result)};
}

/// MaxPool with validation (HIGH-01 fix)
std::vector<Tensor> cpu_maxpool(const std::vector<const Tensor*>& inputs,
                                const OpContext& ctx) {
    validate_input_count(inputs, 1, "MaxPool");
    const Tensor& x = *inputs[0];

    // Security: validate 4D input
    validate_ndim(x, 4, "MaxPool input");

    auto kernel_shape = ctx.node->get_attr<std::vector<int64_t>>("kernel_shape", {2, 2});
    auto strides = ctx.node->get_attr<std::vector<int64_t>>("strides", kernel_shape);
    auto pads = ctx.node->get_attr<std::vector<int64_t>>("pads", {0, 0, 0, 0});

    int64_t N = x.shape()[0];
    int64_t C = x.shape()[1];
    int64_t H = x.shape()[2];
    int64_t W = x.shape()[3];

    int64_t kH = kernel_shape[0];
    int64_t kW = kernel_shape[1];
    int64_t stride_h = strides[0];
    int64_t stride_w = strides[1];
    int64_t pad_h = pads[0];
    int64_t pad_w = pads[1];

    int64_t outH = (H + 2 * pad_h - kH) / stride_h + 1;
    int64_t outW = (W + 2 * pad_w - kW) / stride_w + 1;

    Tensor result({N, C, outH, outW}, x.dtype());

    const float* in = x.data_ptr<float>();
    float* out = result.data_ptr<float>();

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t oh = 0; oh < outH; ++oh) {
                for (int64_t ow = 0; ow < outW; ++ow) {
                    float max_val = -std::numeric_limits<float>::infinity();
                    for (int64_t kh = 0; kh < kH; ++kh) {
                        for (int64_t kw = 0; kw < kW; ++kw) {
                            int64_t ih = oh * stride_h - pad_h + kh;
                            int64_t iw = ow * stride_w - pad_w + kw;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                int64_t idx = ((n * C + c) * H + ih) * W + iw;
                                max_val = std::max(max_val, in[idx]);
                            }
                        }
                    }
                    int64_t out_idx = ((n * C + c) * outH + oh) * outW + ow;
                    out[out_idx] = max_val;
                }
            }
        }
    }

    return {std::move(result)};
}

/// AveragePool with validation (HIGH-01 fix)
std::vector<Tensor> cpu_avgpool(const std::vector<const Tensor*>& inputs,
                                const OpContext& ctx) {
    validate_input_count(inputs, 1, "AveragePool");
    const Tensor& x = *inputs[0];

    // Security: validate 4D input
    validate_ndim(x, 4, "AveragePool input");

    auto kernel_shape = ctx.node->get_attr<std::vector<int64_t>>("kernel_shape", {2, 2});
    auto strides = ctx.node->get_attr<std::vector<int64_t>>("strides", kernel_shape);
    auto pads = ctx.node->get_attr<std::vector<int64_t>>("pads", {0, 0, 0, 0});

    int64_t N = x.shape()[0];
    int64_t C = x.shape()[1];
    int64_t H = x.shape()[2];
    int64_t W = x.shape()[3];

    int64_t kH = kernel_shape[0];
    int64_t kW = kernel_shape[1];
    int64_t stride_h = strides[0];
    int64_t stride_w = strides[1];
    int64_t pad_h = pads[0];
    int64_t pad_w = pads[1];

    int64_t outH = (H + 2 * pad_h - kH) / stride_h + 1;
    int64_t outW = (W + 2 * pad_w - kW) / stride_w + 1;

    Tensor result({N, C, outH, outW}, x.dtype());

    const float* in = x.data_ptr<float>();
    float* out = result.data_ptr<float>();

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t oh = 0; oh < outH; ++oh) {
                for (int64_t ow = 0; ow < outW; ++ow) {
                    float sum = 0;
                    int64_t count = 0;
                    for (int64_t kh = 0; kh < kH; ++kh) {
                        for (int64_t kw = 0; kw < kW; ++kw) {
                            int64_t ih = oh * stride_h - pad_h + kh;
                            int64_t iw = ow * stride_w - pad_w + kw;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                int64_t idx = ((n * C + c) * H + ih) * W + iw;
                                sum += in[idx];
                                count++;
                            }
                        }
                    }
                    int64_t out_idx = ((n * C + c) * outH + oh) * outW + ow;
                    out[out_idx] = count > 0 ? sum / count : 0;
                }
            }
        }
    }

    return {std::move(result)};
}

/// GlobalAveragePool with validation (HIGH-01 fix)
std::vector<Tensor> cpu_global_avgpool(const std::vector<const Tensor*>& inputs,
                                       const OpContext& /*ctx*/) {
    validate_input_count(inputs, 1, "GlobalAveragePool");
    const Tensor& x = *inputs[0];

    // Security: validate 4D input
    validate_ndim(x, 4, "GlobalAveragePool input");

    int64_t N = x.shape()[0];
    int64_t C = x.shape()[1];
    int64_t H = x.shape()[2];
    int64_t W = x.shape()[3];

    // Security: check for division by zero (HIGH-04 related)
    if (H == 0 || W == 0) {
        throw std::invalid_argument("GlobalAveragePool: spatial dimensions cannot be zero");
    }

    Tensor result({N, C, 1, 1}, x.dtype());

    const float* in = x.data_ptr<float>();
    float* out = result.data_ptr<float>();

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            float sum = 0;
            for (int64_t h = 0; h < H; ++h) {
                for (int64_t w = 0; w < W; ++w) {
                    sum += in[((n * C + c) * H + h) * W + w];
                }
            }
            out[n * C + c] = sum / (H * W);
        }
    }

    return {std::move(result)};
}

/// BatchNormalization with validation (HIGH-01 fix)
std::vector<Tensor> cpu_batchnorm(const std::vector<const Tensor*>& inputs,
                                  const OpContext& ctx) {
    validate_input_count(inputs, 5, "BatchNormalization");

    const Tensor& x = *inputs[0];
    const Tensor& scale = *inputs[1];
    const Tensor& bias = *inputs[2];
    const Tensor& mean = *inputs[3];
    const Tensor& var = *inputs[4];

    // Security: validate minimum dimensions
    validate_min_ndim(x, 2, "BatchNormalization input");

    float epsilon = ctx.node->get_attr<float>("epsilon", 1e-5f);

    Tensor result(x.shape(), x.dtype());

    const float* x_data = x.data_ptr<float>();
    const float* scale_data = scale.data_ptr<float>();
    const float* bias_data = bias.data_ptr<float>();
    const float* mean_data = mean.data_ptr<float>();
    const float* var_data = var.data_ptr<float>();
    float* out_data = result.data_ptr<float>();

    int64_t N = x.shape()[0];
    int64_t C = x.shape()[1];
    int64_t spatial = 1;
    for (size_t i = 2; i < x.ndim(); ++i) {
        spatial *= x.shape()[i];
    }

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            float s = scale_data[c];
            float b = bias_data[c];
            float m = mean_data[c];
            float v = var_data[c];
            float inv_std = 1.0f / std::sqrt(v + epsilon);

            for (int64_t i = 0; i < spatial; ++i) {
                int64_t idx = (n * C + c) * spatial + i;
                out_data[idx] = s * (x_data[idx] - m) * inv_std + b;
            }
        }
    }

    return {std::move(result)};
}

/// Dropout with validation
std::vector<Tensor> cpu_dropout(const std::vector<const Tensor*>& inputs,
                                const OpContext& /*ctx*/) {
    validate_input_count(inputs, 1, "Dropout");
    // In inference mode, dropout is identity
    return {inputs[0]->clone()};
}

/// LayerNormalization with validation (HIGH-01 fix)
std::vector<Tensor> cpu_layernorm(const std::vector<const Tensor*>& inputs,
                                  const OpContext& ctx) {
    validate_input_count(inputs, 3, "LayerNormalization");

    const Tensor& x = *inputs[0];
    const Tensor& scale = *inputs[1];
    const Tensor& bias = *inputs[2];

    // Security: validate minimum dimensions
    validate_min_ndim(x, 1, "LayerNormalization input");

    float epsilon = ctx.node->get_attr<float>("epsilon", 1e-5f);
    int64_t axis = ctx.node->get_attr<int64_t>("axis", -1);

    if (axis < 0) axis += static_cast<int64_t>(x.ndim());

    // Security: validate axis
    if (axis < 0 || axis >= static_cast<int64_t>(x.ndim())) {
        throw std::invalid_argument(
            "LayerNormalization: axis " + std::to_string(axis) + " out of range");
    }

    Tensor result(x.shape(), x.dtype());

    const float* x_data = x.data_ptr<float>();
    const float* scale_data = scale.data_ptr<float>();
    const float* bias_data = bias.data_ptr<float>();
    float* out_data = result.data_ptr<float>();

    int64_t outer_size = 1;
    for (int64_t i = 0; i < axis; ++i) {
        outer_size *= x.shape()[i];
    }
    int64_t norm_size = 1;
    for (size_t i = axis; i < x.ndim(); ++i) {
        norm_size *= x.shape()[i];
    }

    for (int64_t o = 0; o < outer_size; ++o) {
        // Calculate mean
        float mean = 0;
        for (int64_t i = 0; i < norm_size; ++i) {
            mean += x_data[o * norm_size + i];
        }
        mean /= norm_size;

        // Calculate variance
        float var = 0;
        for (int64_t i = 0; i < norm_size; ++i) {
            float diff = x_data[o * norm_size + i] - mean;
            var += diff * diff;
        }
        var /= norm_size;

        float inv_std = 1.0f / std::sqrt(var + epsilon);

        // Normalize and apply scale/bias
        for (int64_t i = 0; i < norm_size; ++i) {
            int64_t idx = o * norm_size + i;
            out_data[idx] = scale_data[i] * (x_data[idx] - mean) * inv_std + bias_data[i];
        }
    }

    return {std::move(result)};
}

struct NNOpsRegistrar {
    NNOpsRegistrar() {
        auto& reg = OperatorRegistry::instance();
        reg.register_op("Conv", cpu_conv);
        reg.register_op("MaxPool", cpu_maxpool);
        reg.register_op("AveragePool", cpu_avgpool);
        reg.register_op("GlobalAveragePool", cpu_global_avgpool);
        reg.register_op("BatchNormalization", cpu_batchnorm);
        reg.register_op("Dropout", cpu_dropout);
        reg.register_op("LayerNormalization", cpu_layernorm);
    }
};

static NNOpsRegistrar nn_ops_registrar;

} // anonymous namespace
} // namespace ops
} // namespace pyflame_rt
