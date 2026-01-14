#include "pyflame_rt/registry.hpp"
#include "pyflame_rt/tensor.hpp"
#include "pyflame_rt/errors.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

namespace pyflame_rt {
namespace ops {

namespace {

// ============================================================================
// Validation Helpers
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

/// Validate permutation for transpose (HIGH-06 fix)
inline void validate_permutation(const std::vector<int64_t>& perm, size_t ndim, const char* op_name) {
    // Check size matches
    if (perm.size() != ndim) {
        throw std::invalid_argument(
            std::string(op_name) + ": permutation size (" + std::to_string(perm.size()) +
            ") must match tensor dimensions (" + std::to_string(ndim) + ")");
    }

    // Check all values are in valid range and unique
    std::unordered_set<int64_t> seen;
    for (size_t i = 0; i < perm.size(); ++i) {
        if (perm[i] < 0 || perm[i] >= static_cast<int64_t>(ndim)) {
            throw std::invalid_argument(
                std::string(op_name) + ": permutation value " + std::to_string(perm[i]) +
                " at index " + std::to_string(i) + " is out of range [0, " +
                std::to_string(ndim - 1) + "]");
        }
        if (seen.count(perm[i])) {
            throw std::invalid_argument(
                std::string(op_name) + ": duplicate value " + std::to_string(perm[i]) +
                " in permutation");
        }
        seen.insert(perm[i]);
    }
}

/// Validate axis is in valid range
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
// Tensor Operations
// ============================================================================

/// Reshape with validation (MED-03 fix)
std::vector<Tensor> cpu_reshape(const std::vector<const Tensor*>& inputs,
                                const OpContext& ctx) {
    validate_input_count(inputs, 1, "Reshape");
    const Tensor& x = *inputs[0];

    // Get shape from attribute or second input
    std::vector<int64_t> new_shape;
    if (inputs.size() > 1) {
        const Tensor& shape_tensor = *inputs[1];
        const int64_t* shape_data = shape_tensor.data_ptr<int64_t>();
        for (int64_t i = 0; i < shape_tensor.num_elements(); ++i) {
            new_shape.push_back(shape_data[i]);
        }
    } else {
        new_shape = ctx.node->get_attr<std::vector<int64_t>>("shape", {});
    }

    // Validate: only one -1 allowed, other dims must be positive or 0
    int unknown_count = 0;
    for (size_t i = 0; i < new_shape.size(); ++i) {
        if (new_shape[i] == -1) {
            unknown_count++;
        } else if (new_shape[i] < 0) {
            throw std::invalid_argument(
                "Reshape: invalid dimension " + std::to_string(new_shape[i]) +
                " at index " + std::to_string(i) + " (only -1 allowed for inference)");
        }
    }
    if (unknown_count > 1) {
        throw std::invalid_argument("Reshape: only one dimension can be -1");
    }

    // Handle -1 and 0 dimensions
    int64_t total = x.num_elements();
    int64_t unknown_idx = -1;
    int64_t known_product = 1;

    for (size_t i = 0; i < new_shape.size(); ++i) {
        if (new_shape[i] == -1) {
            unknown_idx = static_cast<int64_t>(i);
        } else if (new_shape[i] == 0) {
            // Security: validate index is in range before accessing source shape
            if (i >= x.ndim()) {
                throw std::invalid_argument(
                    "Reshape: dimension 0 at index " + std::to_string(i) +
                    " but input only has " + std::to_string(x.ndim()) + " dimensions");
            }
            new_shape[i] = x.shape()[i];
            known_product *= new_shape[i];
        } else {
            known_product *= new_shape[i];
        }
    }

    if (unknown_idx >= 0) {
        if (known_product == 0) {
            // MED-03 fix: If known_product is 0 but total is also 0, the -1 dim can be any value
            // Use 0 to maintain zero elements
            if (total == 0) {
                new_shape[unknown_idx] = 0;
            } else {
                throw std::invalid_argument("Reshape: cannot infer dimension when other dimensions multiply to 0");
            }
        } else {
            // MED-03 fix: Check divisibility
            if (total % known_product != 0) {
                throw std::invalid_argument(
                    "Reshape: cannot reshape tensor with " + std::to_string(total) +
                    " elements to shape with inferred dimension (not divisible by " +
                    std::to_string(known_product) + ")");
            }
            new_shape[unknown_idx] = total / known_product;
        }
    } else {
        // MED-03 fix: No -1 dimension, validate that total elements match
        // Security fix MED-03: Use checked_product for overflow detection
        int64_t new_total;
        try {
            new_total = checked_product(new_shape);
        } catch (const std::overflow_error&) {
            throw std::invalid_argument(
                "Reshape: target shape would overflow element count");
        }
        if (new_total != total) {
            throw std::invalid_argument(
                "Reshape: cannot reshape tensor with " + std::to_string(total) +
                " elements to shape with " + std::to_string(new_total) + " elements");
        }
    }

    return {x.reshape(new_shape)};
}

/// Transpose with permutation validation (HIGH-06 fix)
std::vector<Tensor> cpu_transpose(const std::vector<const Tensor*>& inputs,
                                  const OpContext& ctx) {
    validate_input_count(inputs, 1, "Transpose");
    const Tensor& x = *inputs[0];

    // Handle empty tensor
    if (x.ndim() == 0) {
        return {x.clone()};
    }

    auto perm = ctx.node->get_attr<std::vector<int64_t>>("perm", {});

    // Default permutation is reverse
    if (perm.empty()) {
        perm.resize(x.ndim());
        for (size_t i = 0; i < x.ndim(); ++i) {
            perm[i] = static_cast<int64_t>(x.ndim() - 1 - i);
        }
    }

    // Security: validate permutation (HIGH-06)
    validate_permutation(perm, x.ndim(), "Transpose");

    // Calculate new shape
    std::vector<int64_t> new_shape(x.ndim());
    for (size_t i = 0; i < x.ndim(); ++i) {
        new_shape[i] = x.shape()[static_cast<size_t>(perm[i])];
    }

    Tensor result(new_shape, x.dtype());

    // Handle empty result
    if (result.num_elements() == 0) {
        return {std::move(result)};
    }

    // Calculate strides
    std::vector<int64_t> in_strides(x.ndim(), 1);
    std::vector<int64_t> out_strides(x.ndim(), 1);
    for (int i = static_cast<int>(x.ndim()) - 2; i >= 0; --i) {
        in_strides[i] = in_strides[i + 1] * x.shape()[i + 1];
        out_strides[i] = out_strides[i + 1] * new_shape[i + 1];
    }

    const float* in = x.data_ptr<float>();
    float* out = result.data_ptr<float>();

    // Transpose elements
    std::vector<int64_t> indices(x.ndim(), 0);
    for (int64_t i = 0; i < x.num_elements(); ++i) {
        // Calculate source index
        int64_t src_idx = 0;
        for (size_t d = 0; d < x.ndim(); ++d) {
            src_idx += indices[static_cast<size_t>(perm[d])] * in_strides[static_cast<size_t>(perm[d])];
        }

        // Calculate destination index
        int64_t dst_idx = 0;
        for (size_t d = 0; d < x.ndim(); ++d) {
            dst_idx += indices[d] * out_strides[d];
        }

        out[dst_idx] = in[src_idx];

        // Increment indices
        for (int d = static_cast<int>(x.ndim()) - 1; d >= 0; --d) {
            indices[d]++;
            if (indices[d] < new_shape[d]) break;
            indices[d] = 0;
        }
    }

    return {std::move(result)};
}

std::vector<Tensor> cpu_concat(const std::vector<const Tensor*>& inputs,
                               const OpContext& ctx) {
    int64_t axis = ctx.node->get_attr<int64_t>("axis", 0);

    // Handle negative axis
    if (axis < 0) {
        axis = static_cast<int64_t>(inputs[0]->ndim()) + axis;
    }

    // Calculate output shape
    std::vector<int64_t> out_shape = inputs[0]->shape();
    for (size_t i = 1; i < inputs.size(); ++i) {
        out_shape[axis] += inputs[i]->shape()[axis];
    }

    Tensor result(out_shape, inputs[0]->dtype());
    float* out = result.data_ptr<float>();

    // Calculate strides
    int64_t outer_size = 1;
    for (int64_t i = 0; i < axis; ++i) {
        outer_size *= out_shape[i];
    }
    int64_t inner_size = 1;
    for (size_t i = axis + 1; i < out_shape.size(); ++i) {
        inner_size *= out_shape[i];
    }

    // Copy data from each input
    int64_t axis_offset = 0;
    for (const Tensor* input : inputs) {
        const float* in = input->data_ptr<float>();
        int64_t input_axis_size = input->shape()[axis];

        for (int64_t o = 0; o < outer_size; ++o) {
            for (int64_t a = 0; a < input_axis_size; ++a) {
                for (int64_t i = 0; i < inner_size; ++i) {
                    int64_t out_idx = (o * out_shape[axis] + axis_offset + a) * inner_size + i;
                    int64_t in_idx = (o * input_axis_size + a) * inner_size + i;
                    out[out_idx] = in[in_idx];
                }
            }
        }
        axis_offset += input_axis_size;
    }

    return {std::move(result)};
}

std::vector<Tensor> cpu_squeeze(const std::vector<const Tensor*>& inputs,
                                const OpContext& ctx) {
    const Tensor& x = *inputs[0];
    auto axes = ctx.node->get_attr<std::vector<int64_t>>("axes", {});

    std::vector<int64_t> new_shape;
    for (size_t i = 0; i < x.ndim(); ++i) {
        bool should_squeeze = false;
        if (axes.empty()) {
            should_squeeze = (x.shape()[i] == 1);
        } else {
            for (int64_t axis : axes) {
                if (axis < 0) axis += static_cast<int64_t>(x.ndim());
                if (static_cast<size_t>(axis) == i && x.shape()[i] == 1) {
                    should_squeeze = true;
                    break;
                }
            }
        }
        if (!should_squeeze) {
            new_shape.push_back(x.shape()[i]);
        }
    }

    if (new_shape.empty()) {
        new_shape.push_back(1);
    }

    return {x.reshape(new_shape)};
}

/// Unsqueeze with proper axis handling (CRIT-01 fix)
std::vector<Tensor> cpu_unsqueeze(const std::vector<const Tensor*>& inputs,
                                  const OpContext& ctx) {
    validate_input_count(inputs, 1, "Unsqueeze");
    const Tensor& x = *inputs[0];
    auto axes = ctx.node->get_attr<std::vector<int64_t>>("axes", {});

    if (axes.empty()) {
        return {x.clone()};
    }

    // CRIT-01 fix: Calculate final output rank first
    size_t output_rank = x.ndim() + axes.size();

    // Normalize all axes to positive values based on OUTPUT rank (not input rank)
    std::vector<int64_t> normalized_axes;
    normalized_axes.reserve(axes.size());
    for (int64_t axis : axes) {
        if (axis < 0) {
            axis += static_cast<int64_t>(output_rank);
        }
        if (axis < 0 || axis >= static_cast<int64_t>(output_rank)) {
            throw std::invalid_argument(
                "Unsqueeze: axis " + std::to_string(axis) +
                " out of range for output rank " + std::to_string(output_rank));
        }
        normalized_axes.push_back(axis);
    }

    // Check for duplicate axes
    std::vector<int64_t> sorted_axes = normalized_axes;
    std::sort(sorted_axes.begin(), sorted_axes.end());
    for (size_t i = 1; i < sorted_axes.size(); ++i) {
        if (sorted_axes[i] == sorted_axes[i-1]) {
            throw std::invalid_argument(
                "Unsqueeze: duplicate axis " + std::to_string(sorted_axes[i]));
        }
    }

    // Build output shape by inserting 1s at specified positions
    std::vector<int64_t> new_shape(output_rank);
    std::vector<bool> is_new_dim(output_rank, false);
    for (int64_t axis : normalized_axes) {
        is_new_dim[axis] = true;
    }

    size_t src_idx = 0;
    for (size_t i = 0; i < output_rank; ++i) {
        if (is_new_dim[i]) {
            new_shape[i] = 1;
        } else {
            if (src_idx >= x.ndim()) {
                throw std::invalid_argument("Unsqueeze: internal error - source index out of range");
            }
            new_shape[i] = x.shape()[src_idx++];
        }
    }

    return {x.reshape(new_shape)};
}

std::vector<Tensor> cpu_flatten(const std::vector<const Tensor*>& inputs,
                                const OpContext& ctx) {
    const Tensor& x = *inputs[0];
    int64_t axis = ctx.node->get_attr<int64_t>("axis", 1);

    if (axis < 0) axis += static_cast<int64_t>(x.ndim());

    int64_t dim0 = 1;
    int64_t dim1 = 1;

    for (int64_t i = 0; i < axis; ++i) {
        dim0 *= x.shape()[i];
    }
    for (size_t i = axis; i < x.ndim(); ++i) {
        dim1 *= x.shape()[i];
    }

    return {x.reshape({dim0, dim1})};
}

std::vector<Tensor> cpu_constant(const std::vector<const Tensor*>& /*inputs*/,
                                 const OpContext& ctx) {
    // Get value from attributes
    auto value = ctx.node->get_attr<std::vector<float>>("value", {0.0f});
    auto shape = ctx.node->get_attr<std::vector<int64_t>>("shape", {1});

    // Security: validate value is not empty (LOW-04 fix)
    if (value.empty()) {
        throw std::invalid_argument("Constant: value attribute cannot be empty");
    }

    Tensor result(shape, DType::Float32);
    float* out = result.data_ptr<float>();

    // Fill with value
    int64_t n = result.num_elements();
    for (int64_t i = 0; i < n; ++i) {
        out[i] = value[i % value.size()];
    }

    return {std::move(result)};
}

std::vector<Tensor> cpu_identity(const std::vector<const Tensor*>& inputs,
                                 const OpContext& /*ctx*/) {
    validate_input_count(inputs, 1, "Identity");
    return {inputs[0]->clone()};
}

/// Slice with proper n-dimensional implementation (HIGH-05 fix)
std::vector<Tensor> cpu_slice(const std::vector<const Tensor*>& inputs,
                              const OpContext& ctx) {
    validate_input_count(inputs, 1, "Slice");
    const Tensor& x = *inputs[0];

    // Get slice parameters
    auto starts = ctx.node->get_attr<std::vector<int64_t>>("starts", {});
    auto ends = ctx.node->get_attr<std::vector<int64_t>>("ends", {});
    auto axes = ctx.node->get_attr<std::vector<int64_t>>("axes", {});
    auto steps = ctx.node->get_attr<std::vector<int64_t>>("steps", {});

    // Validate required parameters
    if (starts.empty() || ends.empty()) {
        throw std::invalid_argument("Slice: starts and ends must be provided");
    }
    if (starts.size() != ends.size()) {
        throw std::invalid_argument("Slice: starts and ends must have same length");
    }

    // Default axes and steps if not provided
    if (axes.empty()) {
        axes.resize(starts.size());
        std::iota(axes.begin(), axes.end(), 0);
    }
    if (steps.empty()) {
        steps.resize(starts.size(), 1);
    }

    // Validate sizes match
    if (axes.size() != starts.size() || steps.size() != starts.size()) {
        throw std::invalid_argument("Slice: axes, starts, ends, and steps must have same length");
    }

    // Build per-axis slice info (defaults for axes not in the list)
    std::vector<int64_t> slice_starts(x.ndim(), 0);
    std::vector<int64_t> slice_ends(x.ndim());
    std::vector<int64_t> slice_steps(x.ndim(), 1);
    for (size_t i = 0; i < x.ndim(); ++i) {
        slice_ends[i] = x.shape()[i];
    }

    for (size_t i = 0; i < axes.size(); ++i) {
        int64_t axis = axes[i];
        // Validate axis
        if (axis < 0) axis += static_cast<int64_t>(x.ndim());
        if (axis < 0 || axis >= static_cast<int64_t>(x.ndim())) {
            throw std::invalid_argument("Slice: axis " + std::to_string(axes[i]) + " out of range");
        }

        int64_t dim_size = x.shape()[axis];
        int64_t start = starts[i];
        int64_t end = ends[i];
        int64_t step = steps[i];

        // Validate step
        if (step == 0) {
            throw std::invalid_argument("Slice: step cannot be zero");
        }

        // HIGH-05 fix: Properly handle negative steps according to ONNX spec
        // For negative steps, default start is dim_size-1, default end is -dim_size-1
        // We need to handle very large values (INT64_MAX/MIN) as "default"
        const int64_t INT_MAX_THRESHOLD = INT64_MAX / 2;
        const int64_t INT_MIN_THRESHOLD = INT64_MIN / 2;

        if (step > 0) {
            // Positive step: iterate forward from start to end-1
            // Handle default values (very large positive/negative)
            if (start >= INT_MAX_THRESHOLD) start = dim_size;
            if (start <= INT_MIN_THRESHOLD) start = 0;
            if (end >= INT_MAX_THRESHOLD) end = dim_size;
            if (end <= INT_MIN_THRESHOLD) end = 0;

            // Handle negative indices
            if (start < 0) start += dim_size;
            if (end < 0) end += dim_size;

            // Clamp to valid range
            start = std::max(int64_t(0), std::min(start, dim_size));
            end = std::max(int64_t(0), std::min(end, dim_size));

            // Ensure start <= end for positive step
            if (start > end) end = start;
        } else {
            // Negative step: iterate backward from start to end+1
            // Handle default values (very large positive/negative)
            if (start >= INT_MAX_THRESHOLD) start = dim_size - 1;
            if (start <= INT_MIN_THRESHOLD) start = -1;
            if (end >= INT_MAX_THRESHOLD) end = dim_size;
            if (end <= INT_MIN_THRESHOLD) end = -dim_size - 1;

            // Handle negative indices
            if (start < 0) start += dim_size;
            if (end < 0) end += dim_size;

            // Clamp to valid range for reverse iteration
            // start can be at most dim_size - 1
            // end can be -1 (to include index 0)
            start = std::max(int64_t(-1), std::min(start, dim_size - 1));
            end = std::max(int64_t(-1), std::min(end, dim_size - 1));

            // Ensure start >= end for negative step
            if (start < end) start = end;
        }

        slice_starts[axis] = start;
        slice_ends[axis] = end;
        slice_steps[axis] = step;
    }

    // HIGH-05 fix: Calculate output shape considering step direction
    std::vector<int64_t> out_shape(x.ndim());
    for (size_t i = 0; i < x.ndim(); ++i) {
        int64_t start = slice_starts[i];
        int64_t end = slice_ends[i];
        int64_t step = slice_steps[i];

        int64_t length;
        if (step > 0) {
            length = end - start;
        } else {
            // For negative step, we go from start down to end+1
            length = start - end;
        }

        if (length <= 0) {
            out_shape[i] = 0;
        } else {
            int64_t abs_step = std::abs(step);
            out_shape[i] = (length + abs_step - 1) / abs_step;
        }
    }

    Tensor result(out_shape, x.dtype());

    // Handle empty result
    if (result.num_elements() == 0) {
        return {std::move(result)};
    }

    // Calculate input strides
    std::vector<int64_t> in_strides(x.ndim(), 1);
    for (int i = static_cast<int>(x.ndim()) - 2; i >= 0; --i) {
        in_strides[i] = in_strides[i + 1] * x.shape()[i + 1];
    }

    const float* in = x.data_ptr<float>();
    float* out = result.data_ptr<float>();

    // N-dimensional slice copy
    std::vector<int64_t> out_indices(x.ndim(), 0);
    int64_t out_idx = 0;

    while (true) {
        // Calculate input index from output indices and slice parameters
        int64_t in_idx = 0;
        for (size_t d = 0; d < x.ndim(); ++d) {
            int64_t src_coord = slice_starts[d] + out_indices[d] * slice_steps[d];
            in_idx += src_coord * in_strides[d];
        }

        out[out_idx++] = in[in_idx];

        // Increment output indices
        int d = static_cast<int>(x.ndim()) - 1;
        while (d >= 0) {
            out_indices[d]++;
            if (out_indices[d] < out_shape[d]) break;
            out_indices[d] = 0;
            d--;
        }
        if (d < 0) break;  // Done
    }

    return {std::move(result)};
}

struct TensorOpsRegistrar {
    TensorOpsRegistrar() {
        auto& reg = OperatorRegistry::instance();
        reg.register_op("Reshape", cpu_reshape);
        reg.register_op("Transpose", cpu_transpose);
        reg.register_op("Concat", cpu_concat);
        reg.register_op("Squeeze", cpu_squeeze);
        reg.register_op("Unsqueeze", cpu_unsqueeze);
        reg.register_op("Flatten", cpu_flatten);
        reg.register_op("Constant", cpu_constant);
        reg.register_op("Identity", cpu_identity);
        reg.register_op("Slice", cpu_slice);
    }
};

static TensorOpsRegistrar tensor_ops_registrar;

} // anonymous namespace
} // namespace ops
} // namespace pyflame_rt
