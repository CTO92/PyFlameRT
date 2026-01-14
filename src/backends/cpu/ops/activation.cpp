#include "pyflame_rt/registry.hpp"
#include "pyflame_rt/tensor.hpp"
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace pyflame_rt {
namespace ops {

namespace {

// ============================================================================
// Validation Helpers (LOW-01 fix)
// ============================================================================

/// Validate input count for operators
inline void validate_input_count(const std::vector<const Tensor*>& inputs,
                                  size_t min_count, const char* op_name) {
    if (inputs.size() < min_count) {
        throw std::invalid_argument(
            std::string(op_name) + " requires at least " +
            std::to_string(min_count) + " inputs, got " +
            std::to_string(inputs.size()));
    }
}

/// Validate axis is in valid range (LOW-02 fix)
inline int64_t validate_axis(int64_t axis, size_t ndim, const char* op_name) {
    if (axis < 0) {
        axis += static_cast<int64_t>(ndim);
    }
    if (axis < 0 || axis >= static_cast<int64_t>(ndim)) {
        throw std::invalid_argument(
            std::string(op_name) + ": axis " + std::to_string(axis) +
            " is out of range for tensor with " + std::to_string(ndim) + " dimensions");
    }
    return axis;
}

// ============================================================================
// Activation Functions with Input Validation
// ============================================================================

std::vector<Tensor> cpu_relu(const std::vector<const Tensor*>& inputs,
                             const OpContext& /*ctx*/) {
    validate_input_count(inputs, 1, "Relu");
    const Tensor& x = *inputs[0];
    Tensor result(x.shape(), x.dtype());

    const float* in = x.data_ptr<float>();
    float* out = result.data_ptr<float>();
    int64_t n = x.num_elements();

    for (int64_t i = 0; i < n; ++i) {
        out[i] = std::max(0.0f, in[i]);
    }

    return {std::move(result)};
}

std::vector<Tensor> cpu_sigmoid(const std::vector<const Tensor*>& inputs,
                                const OpContext& /*ctx*/) {
    validate_input_count(inputs, 1, "Sigmoid");
    const Tensor& x = *inputs[0];
    Tensor result(x.shape(), x.dtype());

    const float* in = x.data_ptr<float>();
    float* out = result.data_ptr<float>();
    int64_t n = x.num_elements();

    for (int64_t i = 0; i < n; ++i) {
        out[i] = 1.0f / (1.0f + std::exp(-in[i]));
    }

    return {std::move(result)};
}

std::vector<Tensor> cpu_tanh(const std::vector<const Tensor*>& inputs,
                             const OpContext& /*ctx*/) {
    validate_input_count(inputs, 1, "Tanh");
    const Tensor& x = *inputs[0];
    Tensor result(x.shape(), x.dtype());

    const float* in = x.data_ptr<float>();
    float* out = result.data_ptr<float>();
    int64_t n = x.num_elements();

    for (int64_t i = 0; i < n; ++i) {
        out[i] = std::tanh(in[i]);
    }

    return {std::move(result)};
}

std::vector<Tensor> cpu_leaky_relu(const std::vector<const Tensor*>& inputs,
                                   const OpContext& ctx) {
    validate_input_count(inputs, 1, "LeakyRelu");
    const Tensor& x = *inputs[0];
    float alpha = ctx.node->get_attr<float>("alpha", 0.01f);

    Tensor result(x.shape(), x.dtype());

    const float* in = x.data_ptr<float>();
    float* out = result.data_ptr<float>();
    int64_t n = x.num_elements();

    for (int64_t i = 0; i < n; ++i) {
        out[i] = in[i] > 0 ? in[i] : alpha * in[i];
    }

    return {std::move(result)};
}

std::vector<Tensor> cpu_elu(const std::vector<const Tensor*>& inputs,
                            const OpContext& ctx) {
    validate_input_count(inputs, 1, "Elu");
    const Tensor& x = *inputs[0];
    float alpha = ctx.node->get_attr<float>("alpha", 1.0f);

    Tensor result(x.shape(), x.dtype());

    const float* in = x.data_ptr<float>();
    float* out = result.data_ptr<float>();
    int64_t n = x.num_elements();

    for (int64_t i = 0; i < n; ++i) {
        out[i] = in[i] > 0 ? in[i] : alpha * (std::exp(in[i]) - 1.0f);
    }

    return {std::move(result)};
}

std::vector<Tensor> cpu_selu(const std::vector<const Tensor*>& inputs,
                             const OpContext& /*ctx*/) {
    validate_input_count(inputs, 1, "Selu");
    const Tensor& x = *inputs[0];

    // SELU constants
    constexpr float alpha = 1.6732632423543772848170429916717f;
    constexpr float scale = 1.0507009873554804934193349852946f;

    Tensor result(x.shape(), x.dtype());

    const float* in = x.data_ptr<float>();
    float* out = result.data_ptr<float>();
    int64_t n = x.num_elements();

    for (int64_t i = 0; i < n; ++i) {
        float val = in[i];
        out[i] = scale * (val > 0 ? val : alpha * (std::exp(val) - 1.0f));
    }

    return {std::move(result)};
}

std::vector<Tensor> cpu_softmax(const std::vector<const Tensor*>& inputs,
                                const OpContext& ctx) {
    validate_input_count(inputs, 1, "Softmax");
    const Tensor& x = *inputs[0];

    // Security: validate axis range (LOW-02 fix)
    int64_t axis = ctx.node->get_attr<int64_t>("axis", -1);
    axis = validate_axis(axis, x.ndim(), "Softmax");

    Tensor result(x.shape(), x.dtype());

    const float* in = x.data_ptr<float>();
    float* out = result.data_ptr<float>();

    // Calculate strides for the given axis
    int64_t outer_size = 1;
    for (int64_t i = 0; i < axis; ++i) {
        outer_size *= x.shape()[i];
    }
    int64_t axis_size = x.shape()[axis];
    int64_t inner_size = 1;
    for (size_t i = axis + 1; i < x.ndim(); ++i) {
        inner_size *= x.shape()[i];
    }

    // MED-04 fix: Handle empty axis (axis_size == 0)
    // Empty softmax is a valid operation that produces empty output
    if (axis_size == 0) {
        return {std::move(result)};
    }

    for (int64_t o = 0; o < outer_size; ++o) {
        for (int64_t inner = 0; inner < inner_size; ++inner) {
            // Find max for numerical stability
            float max_val = -std::numeric_limits<float>::infinity();
            for (int64_t a = 0; a < axis_size; ++a) {
                int64_t idx = (o * axis_size + a) * inner_size + inner;
                max_val = std::max(max_val, in[idx]);
            }

            // MED-04 fix: Handle all -inf case
            // When all values are -inf, max_val is -inf, and (x - max_val) = NaN
            // In this case, output uniform distribution (1/n for each element)
            if (std::isinf(max_val) && max_val < 0) {
                float uniform_val = 1.0f / static_cast<float>(axis_size);
                for (int64_t a = 0; a < axis_size; ++a) {
                    int64_t idx = (o * axis_size + a) * inner_size + inner;
                    out[idx] = uniform_val;
                }
                continue;
            }

            // Compute exp and sum
            float sum = 0;
            for (int64_t a = 0; a < axis_size; ++a) {
                int64_t idx = (o * axis_size + a) * inner_size + inner;
                float exp_val = std::exp(in[idx] - max_val);
                out[idx] = exp_val;
                sum += exp_val;
            }

            // MED-04 fix: Handle sum == 0 (shouldn't happen if max_val check works, but be safe)
            if (sum == 0.0f) {
                float uniform_val = 1.0f / static_cast<float>(axis_size);
                for (int64_t a = 0; a < axis_size; ++a) {
                    int64_t idx = (o * axis_size + a) * inner_size + inner;
                    out[idx] = uniform_val;
                }
                continue;
            }

            // Normalize
            for (int64_t a = 0; a < axis_size; ++a) {
                int64_t idx = (o * axis_size + a) * inner_size + inner;
                out[idx] /= sum;
            }
        }
    }

    return {std::move(result)};
}

std::vector<Tensor> cpu_gelu(const std::vector<const Tensor*>& inputs,
                             const OpContext& /*ctx*/) {
    validate_input_count(inputs, 1, "Gelu");
    const Tensor& x = *inputs[0];
    Tensor result(x.shape(), x.dtype());

    const float* in = x.data_ptr<float>();
    float* out = result.data_ptr<float>();
    int64_t n = x.num_elements();

    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    constexpr float sqrt_2_over_pi = 0.7978845608028654f;

    for (int64_t i = 0; i < n; ++i) {
        float val = in[i];
        float inner = sqrt_2_over_pi * (val + 0.044715f * val * val * val);
        out[i] = 0.5f * val * (1.0f + std::tanh(inner));
    }

    return {std::move(result)};
}

std::vector<Tensor> cpu_hardswish(const std::vector<const Tensor*>& inputs,
                                   const OpContext& /*ctx*/) {
    validate_input_count(inputs, 1, "HardSwish");
    const Tensor& x = *inputs[0];
    Tensor result(x.shape(), x.dtype());

    const float* in = x.data_ptr<float>();
    float* out = result.data_ptr<float>();
    int64_t n = x.num_elements();

    for (int64_t i = 0; i < n; ++i) {
        float val = in[i];
        if (val <= -3.0f) {
            out[i] = 0.0f;
        } else if (val >= 3.0f) {
            out[i] = val;
        } else {
            out[i] = val * (val + 3.0f) / 6.0f;
        }
    }

    return {std::move(result)};
}

struct ActivationOpsRegistrar {
    ActivationOpsRegistrar() {
        auto& reg = OperatorRegistry::instance();
        reg.register_op("Relu", cpu_relu);
        reg.register_op("Sigmoid", cpu_sigmoid);
        reg.register_op("Tanh", cpu_tanh);
        reg.register_op("LeakyRelu", cpu_leaky_relu);
        reg.register_op("Elu", cpu_elu);
        reg.register_op("Selu", cpu_selu);
        reg.register_op("Softmax", cpu_softmax);
        reg.register_op("Gelu", cpu_gelu);
        reg.register_op("HardSwish", cpu_hardswish);
    }
};

static ActivationOpsRegistrar activation_ops_registrar;

} // anonymous namespace
} // namespace ops
} // namespace pyflame_rt
