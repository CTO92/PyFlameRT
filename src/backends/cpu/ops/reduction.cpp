#include "pyflame_rt/registry.hpp"
#include "pyflame_rt/tensor.hpp"
#include <cmath>
#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace pyflame_rt {
namespace ops {

namespace {

// ============================================================================
// Validation Helpers (LOW-01, LOW-03 fixes)
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

/// Validate and normalize axes for reduction operations (LOW-03 fix)
inline std::vector<int64_t> validate_axes(
    const std::vector<int64_t>& axes, size_t ndim, const char* op_name)
{
    std::vector<int64_t> normalized;
    normalized.reserve(axes.size());

    for (size_t i = 0; i < axes.size(); ++i) {
        int64_t axis = axes[i];
        // Handle negative axis
        if (axis < 0) {
            axis += static_cast<int64_t>(ndim);
        }
        // Validate range
        if (axis < 0 || axis >= static_cast<int64_t>(ndim)) {
            throw std::invalid_argument(
                std::string(op_name) + ": axis " + std::to_string(axes[i]) +
                " is out of range for tensor with " + std::to_string(ndim) + " dimensions");
        }
        normalized.push_back(axis);
    }
    return normalized;
}

/// Validate single axis and return normalized value (LOW-03 fix)
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
// Reduction Operations
// ============================================================================

// Helper to calculate reduced shape
std::vector<int64_t> get_reduced_shape(
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& axes,
    bool keepdims)
{
    std::vector<int64_t> out_shape;

    for (size_t i = 0; i < input_shape.size(); ++i) {
        bool is_reduced = false;
        for (int64_t axis : axes) {
            int64_t a = axis < 0 ? axis + static_cast<int64_t>(input_shape.size()) : axis;
            if (static_cast<size_t>(a) == i) {
                is_reduced = true;
                break;
            }
        }

        if (is_reduced) {
            if (keepdims) {
                out_shape.push_back(1);
            }
        } else {
            out_shape.push_back(input_shape[i]);
        }
    }

    if (out_shape.empty()) {
        out_shape.push_back(1);
    }

    return out_shape;
}

std::vector<Tensor> cpu_reduce_sum(const std::vector<const Tensor*>& inputs,
                                   const OpContext& ctx) {
    validate_input_count(inputs, 1, "ReduceSum");
    const Tensor& x = *inputs[0];
    auto axes = ctx.node->get_attr<std::vector<int64_t>>("axes", {});
    int64_t keepdims = ctx.node->get_attr<int64_t>("keepdims", 1);

    // Default: reduce all axes
    if (axes.empty()) {
        axes.resize(x.ndim());
        std::iota(axes.begin(), axes.end(), 0);
    } else {
        // Security: validate axes (LOW-03 fix)
        axes = validate_axes(axes, x.ndim(), "ReduceSum");
    }

    auto out_shape = get_reduced_shape(x.shape(), axes, keepdims != 0);
    Tensor result(out_shape, x.dtype());
    result.zero();

    const float* in = x.data_ptr<float>();
    float* out = result.data_ptr<float>();

    // Simple implementation for common case: reduce all
    if (axes.size() == x.ndim()) {
        float sum = 0;
        for (int64_t i = 0; i < x.num_elements(); ++i) {
            sum += in[i];
        }
        out[0] = sum;
    } else {
        // General case - iterate through all elements
        int64_t n = x.num_elements();
        std::vector<int64_t> indices(x.ndim(), 0);

        for (int64_t i = 0; i < n; ++i) {
            // Calculate output index
            std::vector<int64_t> out_indices;
            for (size_t d = 0; d < x.ndim(); ++d) {
                bool is_reduced = false;
                for (int64_t axis : axes) {
                    int64_t a = axis < 0 ? axis + static_cast<int64_t>(x.ndim()) : axis;
                    if (static_cast<size_t>(a) == d) {
                        is_reduced = true;
                        break;
                    }
                }
                if (!is_reduced || keepdims) {
                    out_indices.push_back(is_reduced ? 0 : indices[d]);
                }
            }

            // Calculate flat output index
            int64_t out_idx = 0;
            int64_t stride = 1;
            for (int d = static_cast<int>(out_shape.size()) - 1; d >= 0; --d) {
                out_idx += out_indices[d] * stride;
                stride *= out_shape[d];
            }

            out[out_idx] += in[i];

            // Increment indices
            for (int d = static_cast<int>(x.ndim()) - 1; d >= 0; --d) {
                indices[d]++;
                if (indices[d] < x.shape()[d]) break;
                indices[d] = 0;
            }
        }
    }

    return {std::move(result)};
}

std::vector<Tensor> cpu_reduce_mean(const std::vector<const Tensor*>& inputs,
                                    const OpContext& ctx) {
    validate_input_count(inputs, 1, "ReduceMean");
    const Tensor& x = *inputs[0];
    auto axes = ctx.node->get_attr<std::vector<int64_t>>("axes", {});
    // Note: axes validation happens in cpu_reduce_sum which we call
    int64_t keepdims = ctx.node->get_attr<int64_t>("keepdims", 1);

    // First compute sum
    auto sum_result = cpu_reduce_sum(inputs, ctx);
    Tensor& result = sum_result[0];

    // Calculate divisor
    int64_t count = 1;
    if (axes.empty()) {
        count = x.num_elements();
    } else {
        for (int64_t axis : axes) {
            int64_t a = axis < 0 ? axis + static_cast<int64_t>(x.ndim()) : axis;
            count *= x.shape()[a];
        }
    }

    // MED-02 fix: Check for division by zero
    // This can happen with empty tensors or dimensions of size 0
    if (count == 0) {
        // Following IEEE 754 / NumPy behavior: 0/0 = NaN
        float* out = result.data_ptr<float>();
        for (int64_t i = 0; i < result.num_elements(); ++i) {
            out[i] = std::numeric_limits<float>::quiet_NaN();
        }
        return {std::move(result)};
    }

    // Divide by count
    float* out = result.data_ptr<float>();
    for (int64_t i = 0; i < result.num_elements(); ++i) {
        out[i] /= static_cast<float>(count);
    }

    return {std::move(result)};
}

std::vector<Tensor> cpu_reduce_max(const std::vector<const Tensor*>& inputs,
                                   const OpContext& ctx) {
    validate_input_count(inputs, 1, "ReduceMax");
    const Tensor& x = *inputs[0];
    auto axes = ctx.node->get_attr<std::vector<int64_t>>("axes", {});
    int64_t keepdims = ctx.node->get_attr<int64_t>("keepdims", 1);

    if (axes.empty()) {
        axes.resize(x.ndim());
        std::iota(axes.begin(), axes.end(), 0);
    } else {
        // Security: validate axes (LOW-03 fix)
        axes = validate_axes(axes, x.ndim(), "ReduceMax");
    }

    auto out_shape = get_reduced_shape(x.shape(), axes, keepdims != 0);
    Tensor result(out_shape, x.dtype());

    const float* in = x.data_ptr<float>();
    float* out = result.data_ptr<float>();

    // Initialize with -inf
    for (int64_t i = 0; i < result.num_elements(); ++i) {
        out[i] = -std::numeric_limits<float>::infinity();
    }

    // Reduce all case
    if (axes.size() == x.ndim()) {
        float max_val = -std::numeric_limits<float>::infinity();
        for (int64_t i = 0; i < x.num_elements(); ++i) {
            max_val = std::max(max_val, in[i]);
        }
        out[0] = max_val;
    } else {
        // Security fix CRIT-01: Proper n-dimensional index mapping instead of naive modulo
        int64_t n = x.num_elements();
        std::vector<int64_t> indices(x.ndim(), 0);

        for (int64_t i = 0; i < n; ++i) {
            // Calculate output index by projecting input indices onto non-reduced dimensions
            std::vector<int64_t> out_indices;
            for (size_t d = 0; d < x.ndim(); ++d) {
                bool is_reduced = false;
                for (int64_t axis : axes) {
                    int64_t a = axis < 0 ? axis + static_cast<int64_t>(x.ndim()) : axis;
                    if (static_cast<size_t>(a) == d) {
                        is_reduced = true;
                        break;
                    }
                }
                if (!is_reduced || keepdims) {
                    out_indices.push_back(is_reduced ? 0 : indices[d]);
                }
            }

            // Calculate flat output index using strides
            int64_t out_idx = 0;
            int64_t stride = 1;
            for (int d = static_cast<int>(out_shape.size()) - 1; d >= 0; --d) {
                out_idx += out_indices[d] * stride;
                stride *= out_shape[d];
            }

            // Bounds check before access
            if (out_idx >= 0 && out_idx < result.num_elements()) {
                out[out_idx] = std::max(out[out_idx], in[i]);
            }

            // Increment indices
            for (int d = static_cast<int>(x.ndim()) - 1; d >= 0; --d) {
                indices[d]++;
                if (indices[d] < x.shape()[d]) break;
                indices[d] = 0;
            }
        }
    }

    return {std::move(result)};
}

std::vector<Tensor> cpu_reduce_min(const std::vector<const Tensor*>& inputs,
                                   const OpContext& ctx) {
    validate_input_count(inputs, 1, "ReduceMin");
    const Tensor& x = *inputs[0];
    auto axes = ctx.node->get_attr<std::vector<int64_t>>("axes", {});
    int64_t keepdims = ctx.node->get_attr<int64_t>("keepdims", 1);

    if (axes.empty()) {
        axes.resize(x.ndim());
        std::iota(axes.begin(), axes.end(), 0);
    } else {
        // Security: validate axes (LOW-03 fix)
        axes = validate_axes(axes, x.ndim(), "ReduceMin");
    }

    auto out_shape = get_reduced_shape(x.shape(), axes, keepdims != 0);
    Tensor result(out_shape, x.dtype());

    const float* in = x.data_ptr<float>();
    float* out = result.data_ptr<float>();

    // Initialize with +inf
    for (int64_t i = 0; i < result.num_elements(); ++i) {
        out[i] = std::numeric_limits<float>::infinity();
    }

    if (axes.size() == x.ndim()) {
        float min_val = std::numeric_limits<float>::infinity();
        for (int64_t i = 0; i < x.num_elements(); ++i) {
            min_val = std::min(min_val, in[i]);
        }
        out[0] = min_val;
    } else {
        // Security fix CRIT-01: Proper n-dimensional index mapping instead of naive modulo
        int64_t n = x.num_elements();
        std::vector<int64_t> indices(x.ndim(), 0);

        for (int64_t i = 0; i < n; ++i) {
            // Calculate output index by projecting input indices onto non-reduced dimensions
            std::vector<int64_t> out_indices;
            for (size_t d = 0; d < x.ndim(); ++d) {
                bool is_reduced = false;
                for (int64_t axis : axes) {
                    int64_t a = axis < 0 ? axis + static_cast<int64_t>(x.ndim()) : axis;
                    if (static_cast<size_t>(a) == d) {
                        is_reduced = true;
                        break;
                    }
                }
                if (!is_reduced || keepdims) {
                    out_indices.push_back(is_reduced ? 0 : indices[d]);
                }
            }

            // Calculate flat output index using strides
            int64_t out_idx = 0;
            int64_t stride = 1;
            for (int d = static_cast<int>(out_shape.size()) - 1; d >= 0; --d) {
                out_idx += out_indices[d] * stride;
                stride *= out_shape[d];
            }

            // Bounds check before access
            if (out_idx >= 0 && out_idx < result.num_elements()) {
                out[out_idx] = std::min(out[out_idx], in[i]);
            }

            // Increment indices
            for (int d = static_cast<int>(x.ndim()) - 1; d >= 0; --d) {
                indices[d]++;
                if (indices[d] < x.shape()[d]) break;
                indices[d] = 0;
            }
        }
    }

    return {std::move(result)};
}

std::vector<Tensor> cpu_argmax(const std::vector<const Tensor*>& inputs,
                               const OpContext& ctx) {
    validate_input_count(inputs, 1, "ArgMax");
    const Tensor& x = *inputs[0];
    int64_t axis = ctx.node->get_attr<int64_t>("axis", 0);
    int64_t keepdims = ctx.node->get_attr<int64_t>("keepdims", 1);

    // Security: validate axis (LOW-03 fix)
    axis = validate_axis(axis, x.ndim(), "ArgMax");

    std::vector<int64_t> out_shape;
    for (size_t i = 0; i < x.ndim(); ++i) {
        if (static_cast<int64_t>(i) == axis) {
            if (keepdims) out_shape.push_back(1);
        } else {
            out_shape.push_back(x.shape()[i]);
        }
    }
    if (out_shape.empty()) out_shape.push_back(1);

    Tensor result(out_shape, DType::Int64);

    const float* in = x.data_ptr<float>();
    int64_t* out = result.data_ptr<int64_t>();

    int64_t outer_size = 1;
    for (int64_t i = 0; i < axis; ++i) {
        outer_size *= x.shape()[i];
    }
    int64_t axis_size = x.shape()[axis];
    int64_t inner_size = 1;
    for (size_t i = axis + 1; i < x.ndim(); ++i) {
        inner_size *= x.shape()[i];
    }

    for (int64_t o = 0; o < outer_size; ++o) {
        for (int64_t inner = 0; inner < inner_size; ++inner) {
            float max_val = -std::numeric_limits<float>::infinity();
            int64_t max_idx = 0;
            for (int64_t a = 0; a < axis_size; ++a) {
                int64_t idx = (o * axis_size + a) * inner_size + inner;
                if (in[idx] > max_val) {
                    max_val = in[idx];
                    max_idx = a;
                }
            }
            out[o * inner_size + inner] = max_idx;
        }
    }

    return {std::move(result)};
}

struct ReductionOpsRegistrar {
    ReductionOpsRegistrar() {
        auto& reg = OperatorRegistry::instance();
        reg.register_op("ReduceSum", cpu_reduce_sum);
        reg.register_op("ReduceMean", cpu_reduce_mean);
        reg.register_op("ReduceMax", cpu_reduce_max);
        reg.register_op("ReduceMin", cpu_reduce_min);
        reg.register_op("ArgMax", cpu_argmax);
    }
};

static ReductionOpsRegistrar reduction_ops_registrar;

} // anonymous namespace
} // namespace ops
} // namespace pyflame_rt
