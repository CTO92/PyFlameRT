#include "pyflame_rt/registry.hpp"
#include "pyflame_rt/tensor.hpp"
#include "pyflame_rt/errors.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace pyflame_rt {
namespace ops {

namespace {

// ============================================================================
// Security Validation Helpers
// ============================================================================

/// Validate that inputs vector has at least min_count elements
inline void validate_input_count(const std::vector<const Tensor*>& inputs,
                                  size_t min_count, const char* op_name) {
    if (inputs.size() < min_count) {
        throw std::invalid_argument(
            std::string(op_name) + " requires at least " +
            std::to_string(min_count) + " inputs, got " +
            std::to_string(inputs.size()));
    }
}

/// Validate that tensor has at least min_dims dimensions
inline void validate_min_dims(const Tensor& t, size_t min_dims, const char* name) {
    if (t.ndim() < min_dims) {
        throw std::invalid_argument(
            std::string(name) + " requires at least " +
            std::to_string(min_dims) + " dimensions, got " +
            std::to_string(t.ndim()));
    }
}

/// Check if two shapes are broadcastable (HIGH-02 fix)
inline bool are_broadcastable(const std::vector<int64_t>& a,
                               const std::vector<int64_t>& b) {
    size_t max_dims = std::max(a.size(), b.size());
    for (size_t i = 0; i < max_dims; ++i) {
        int64_t dim_a = (i < a.size()) ? a[a.size() - 1 - i] : 1;
        int64_t dim_b = (i < b.size()) ? b[b.size() - 1 - i] : 1;
        // Dimensions are compatible if they are equal or one of them is 1
        if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
            return false;
        }
    }
    return true;
}

/// Calculate broadcast output shape
inline std::vector<int64_t> broadcast_shape(const std::vector<int64_t>& a,
                                             const std::vector<int64_t>& b) {
    size_t max_dims = std::max(a.size(), b.size());
    std::vector<int64_t> result(max_dims);
    for (size_t i = 0; i < max_dims; ++i) {
        int64_t dim_a = (i < a.size()) ? a[a.size() - 1 - i] : 1;
        int64_t dim_b = (i < b.size()) ? b[b.size() - 1 - i] : 1;
        result[max_dims - 1 - i] = std::max(dim_a, dim_b);
    }
    return result;
}

/// Compute strides for a shape (row-major order) with overflow protection (HIGH-02 fix)
inline std::vector<int64_t> compute_strides(const std::vector<int64_t>& shape) {
    std::vector<int64_t> strides(shape.size());
    if (shape.empty()) return strides;

    int64_t stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        strides[i] = stride;
        // Security fix HIGH-02: Use checked_multiply to detect overflow
        stride = checked_multiply(stride, shape[i]);
    }
    return strides;
}

/// Compute broadcast strides - stride is 0 for broadcast dimensions (HIGH-02 fix)
/// This properly handles n-dimensional broadcasting by setting stride to 0
/// for dimensions where the input has size 1 (broadcast dimensions)
inline std::vector<int64_t> compute_broadcast_strides(
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& output_shape)
{
    size_t out_dims = output_shape.size();
    size_t in_dims = input_shape.size();
    std::vector<int64_t> strides(out_dims, 0);

    // Compute input strides
    std::vector<int64_t> input_strides = compute_strides(input_shape);

    // Map input dimensions to output dimensions (right-aligned)
    for (size_t i = 0; i < in_dims; ++i) {
        size_t out_idx = out_dims - in_dims + i;
        // If input dimension is 1 and output is larger, it's broadcast (stride=0)
        // Otherwise, use the actual stride
        if (input_shape[i] == output_shape[out_idx]) {
            strides[out_idx] = input_strides[i];
        } else if (input_shape[i] == 1) {
            strides[out_idx] = 0;  // Broadcast: repeat this element
        } else {
            // HIGH-02 fix: Defensive throw for incompatible shapes
            // This should be caught by are_broadcastable() but we add safety check
            throw std::invalid_argument(
                "compute_broadcast_strides: incompatible shapes at dimension " +
                std::to_string(i) + " (input: " + std::to_string(input_shape[i]) +
                ", output: " + std::to_string(output_shape[out_idx]) + ")");
        }
    }

    return strides;
}

/// Convert linear index to multi-dimensional index
inline void linear_to_ndindex(int64_t linear, const std::vector<int64_t>& shape,
                               std::vector<int64_t>& out_idx) {
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        out_idx[i] = linear % shape[i];
        linear /= shape[i];
    }
}

/// Compute linear index from multi-dimensional index using strides
inline int64_t ndindex_to_linear(const std::vector<int64_t>& idx,
                                  const std::vector<int64_t>& strides) {
    int64_t result = 0;
    for (size_t i = 0; i < idx.size(); ++i) {
        result += idx[i] * strides[i];
    }
    return result;
}

// ============================================================================
// Binary Elementwise Operations with Validation
// ============================================================================

/// Helper for broadcasting with validation (HIGH-02 fix)
/// Uses proper n-dimensional stride-based broadcasting instead of naive modulo
template<typename BinaryOp>
std::vector<Tensor> binary_elementwise(
    const std::vector<const Tensor*>& inputs,
    const OpContext& /*ctx*/,
    BinaryOp op,
    const char* op_name = "binary_op")
{
    // Security: validate input count
    validate_input_count(inputs, 2, op_name);

    const Tensor& a = *inputs[0];
    const Tensor& b = *inputs[1];

    // Security: validate shapes are broadcastable (HIGH-02)
    if (!are_broadcastable(a.shape(), b.shape())) {
        throw std::invalid_argument(
            std::string(op_name) + ": shapes are not broadcastable");
    }

    // Calculate output shape
    std::vector<int64_t> out_shape = broadcast_shape(a.shape(), b.shape());

    Tensor result(out_shape, a.dtype());

    const float* a_data = a.data_ptr<float>();
    const float* b_data = b.data_ptr<float>();
    float* out_data = result.data_ptr<float>();

    int64_t n = result.num_elements();

    // Safety check for empty tensors
    if (a.num_elements() == 0 || b.num_elements() == 0) {
        return {std::move(result)};
    }

    // Fast path: shapes are identical (no broadcasting needed)
    if (a.shape() == b.shape()) {
        for (int64_t i = 0; i < n; ++i) {
            out_data[i] = op(a_data[i], b_data[i]);
        }
        return {std::move(result)};
    }

    // Fast path: one tensor is scalar
    if (a.num_elements() == 1) {
        float a_val = a_data[0];
        for (int64_t i = 0; i < n; ++i) {
            out_data[i] = op(a_val, b_data[i]);
        }
        return {std::move(result)};
    }
    if (b.num_elements() == 1) {
        float b_val = b_data[0];
        for (int64_t i = 0; i < n; ++i) {
            out_data[i] = op(a_data[i], b_val);
        }
        return {std::move(result)};
    }

    // General case: proper n-dimensional broadcasting with strides (HIGH-02 fix)
    // Compute broadcast strides for both inputs
    std::vector<int64_t> a_strides = compute_broadcast_strides(a.shape(), out_shape);
    std::vector<int64_t> b_strides = compute_broadcast_strides(b.shape(), out_shape);

    // Pre-allocate index buffer
    std::vector<int64_t> nd_idx(out_shape.size());

    for (int64_t i = 0; i < n; ++i) {
        // Convert linear output index to n-dimensional index
        linear_to_ndindex(i, out_shape, nd_idx);

        // Compute input indices using broadcast strides
        int64_t a_idx = ndindex_to_linear(nd_idx, a_strides);
        int64_t b_idx = ndindex_to_linear(nd_idx, b_strides);

        out_data[i] = op(a_data[a_idx], b_data[b_idx]);
    }

    return {std::move(result)};
}

std::vector<Tensor> cpu_add(const std::vector<const Tensor*>& inputs,
                            const OpContext& ctx) {
    return binary_elementwise(inputs, ctx,
        [](float a, float b) { return a + b; }, "Add");
}

std::vector<Tensor> cpu_sub(const std::vector<const Tensor*>& inputs,
                            const OpContext& ctx) {
    return binary_elementwise(inputs, ctx,
        [](float a, float b) { return a - b; }, "Sub");
}

std::vector<Tensor> cpu_mul(const std::vector<const Tensor*>& inputs,
                            const OpContext& ctx) {
    return binary_elementwise(inputs, ctx,
        [](float a, float b) { return a * b; }, "Mul");
}

/// Division operator with optional strict zero check (HIGH-04 fix)
std::vector<Tensor> cpu_div(const std::vector<const Tensor*>& inputs,
                            const OpContext& ctx) {
    // Security: validate input count first
    validate_input_count(inputs, 2, "Div");

    // HIGH-04 fix: In strict math mode, check for division by zero
    if (ctx.strict_math_mode) {
        const Tensor& divisor = *inputs[1];
        const float* b_data = divisor.data_ptr<float>();
        int64_t n = divisor.num_elements();

        for (int64_t i = 0; i < n; ++i) {
            if (b_data[i] == 0.0f) {
                throw std::domain_error(
                    "Div: division by zero detected at index " +
                    std::to_string(i) + " (strict_math_mode enabled)");
            }
        }
    }

    // Note: Without strict mode, division by zero produces inf/-inf/nan
    // which is defined IEEE 754 behavior
    return binary_elementwise(inputs, ctx,
        [](float a, float b) { return a / b; }, "Div");
}

/// Matrix multiplication with validation (HIGH-03 fix)
std::vector<Tensor> cpu_matmul(const std::vector<const Tensor*>& inputs,
                               const OpContext& /*ctx*/) {
    // Security: validate input count (HIGH-03)
    validate_input_count(inputs, 2, "MatMul");

    const Tensor& a = *inputs[0];
    const Tensor& b = *inputs[1];

    // Security: validate minimum dimensions (HIGH-03)
    validate_min_dims(a, 2, "MatMul input A");
    validate_min_dims(b, 2, "MatMul input B");

    // Get matrix dimensions
    int64_t M = a.shape()[a.ndim() - 2];
    int64_t K_a = a.shape()[a.ndim() - 1];
    int64_t K_b = b.shape()[b.ndim() - 2];
    int64_t N = b.shape()[b.ndim() - 1];

    // Security: validate inner dimensions match (HIGH-03)
    if (K_a != K_b) {
        throw std::invalid_argument(
            "MatMul: incompatible dimensions - A has inner dim " +
            std::to_string(K_a) + ", B has inner dim " + std::to_string(K_b));
    }
    int64_t K = K_a;

    // HIGH-03 fix: Extract and validate batch dimensions
    std::vector<int64_t> a_batch_dims(a.shape().begin(), a.shape().end() - 2);
    std::vector<int64_t> b_batch_dims(b.shape().begin(), b.shape().end() - 2);

    // Validate batch dimensions are broadcastable
    if (!are_broadcastable(a_batch_dims, b_batch_dims)) {
        throw std::invalid_argument(
            "MatMul: batch dimensions are not broadcastable");
    }

    // Compute broadcast batch shape
    std::vector<int64_t> batch_shape = broadcast_shape(a_batch_dims, b_batch_dims);

    // Build output shape = broadcast_batch_shape + [M, N]
    std::vector<int64_t> out_shape = batch_shape;
    out_shape.push_back(M);
    out_shape.push_back(N);

    Tensor result(out_shape, a.dtype());
    result.zero();

    const float* a_data = a.data_ptr<float>();
    const float* b_data = b.data_ptr<float>();
    float* out_data = result.data_ptr<float>();

    // Calculate total batch size
    int64_t total_batch = 1;
    for (int64_t dim : batch_shape) {
        total_batch *= dim;
    }

    // If no batch dimensions, simple 2D matmul
    if (batch_shape.empty()) {
        for (int64_t i = 0; i < M; ++i) {
            for (int64_t k = 0; k < K; ++k) {
                float a_ik = a_data[i * K + k];
                for (int64_t j = 0; j < N; ++j) {
                    out_data[i * N + j] += a_ik * b_data[k * N + j];
                }
            }
        }
        return {std::move(result)};
    }

    // HIGH-03 fix: Compute broadcast strides for batch dimensions
    std::vector<int64_t> a_batch_strides = compute_broadcast_strides(a_batch_dims, batch_shape);
    std::vector<int64_t> b_batch_strides = compute_broadcast_strides(b_batch_dims, batch_shape);

    // Size of one matrix in each input
    int64_t a_matrix_size = M * K;
    int64_t b_matrix_size = K * N;
    int64_t out_matrix_size = M * N;

    // Pre-allocate batch index buffer
    std::vector<int64_t> batch_idx(batch_shape.size());

    // Batched matrix multiplication with proper broadcasting
    for (int64_t n = 0; n < total_batch; ++n) {
        // Convert linear batch index to n-dimensional batch index
        linear_to_ndindex(n, batch_shape, batch_idx);

        // Compute source batch indices using broadcast strides
        int64_t a_batch_offset = ndindex_to_linear(batch_idx, a_batch_strides);
        int64_t b_batch_offset = ndindex_to_linear(batch_idx, b_batch_strides);

        const float* a_batch = a_data + a_batch_offset * a_matrix_size;
        const float* b_batch = b_data + b_batch_offset * b_matrix_size;
        float* out_batch = out_data + n * out_matrix_size;

        for (int64_t i = 0; i < M; ++i) {
            for (int64_t k = 0; k < K; ++k) {
                float a_ik = a_batch[i * K + k];
                for (int64_t j = 0; j < N; ++j) {
                    out_batch[i * N + j] += a_ik * b_batch[k * N + j];
                }
            }
        }
    }

    return {std::move(result)};
}

/// GEMM with validation
std::vector<Tensor> cpu_gemm(const std::vector<const Tensor*>& inputs,
                             const OpContext& ctx) {
    // Security: validate input count
    validate_input_count(inputs, 2, "Gemm");

    const Tensor& A = *inputs[0];
    const Tensor& B = *inputs[1];
    const Tensor* C = inputs.size() > 2 ? inputs[2] : nullptr;

    // Security: validate dimensions
    validate_min_dims(A, 2, "Gemm input A");
    validate_min_dims(B, 2, "Gemm input B");

    float alpha = ctx.node->get_attr<float>("alpha", 1.0f);
    float beta = ctx.node->get_attr<float>("beta", 1.0f);
    int64_t trans_a = ctx.node->get_attr<int64_t>("transA", 0);
    int64_t trans_b = ctx.node->get_attr<int64_t>("transB", 0);

    int64_t M = trans_a ? A.shape()[1] : A.shape()[0];
    int64_t K_a = trans_a ? A.shape()[0] : A.shape()[1];
    int64_t K_b = trans_b ? B.shape()[1] : B.shape()[0];
    int64_t N = trans_b ? B.shape()[0] : B.shape()[1];

    // Security: validate inner dimensions match
    if (K_a != K_b) {
        throw std::invalid_argument(
            "Gemm: incompatible dimensions - A has K=" + std::to_string(K_a) +
            ", B has K=" + std::to_string(K_b));
    }
    int64_t K = K_a;

    Tensor result({M, N}, A.dtype());
    result.zero();

    const float* a_data = A.data_ptr<float>();
    const float* b_data = B.data_ptr<float>();
    float* out_data = result.data_ptr<float>();

    // Matrix multiplication with optional transpose
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t k = 0; k < K; ++k) {
            float a_ik = trans_a ? a_data[k * M + i] : a_data[i * K + k];
            for (int64_t j = 0; j < N; ++j) {
                float b_kj = trans_b ? b_data[j * K + k] : b_data[k * N + j];
                out_data[i * N + j] += alpha * a_ik * b_kj;
            }
        }
    }

    // Add bias if present
    if (C) {
        const float* c_data = C->data_ptr<float>();
        int64_t c_size = C->num_elements();
        if (c_size > 0) {
            for (int64_t i = 0; i < M * N; ++i) {
                out_data[i] += beta * c_data[i % c_size];
            }
        }
    }

    return {std::move(result)};
}

std::vector<Tensor> cpu_sqrt(const std::vector<const Tensor*>& inputs,
                             const OpContext& /*ctx*/) {
    validate_input_count(inputs, 1, "Sqrt");  // LOW-01 fix
    const Tensor& x = *inputs[0];
    Tensor result(x.shape(), x.dtype());

    const float* in = x.data_ptr<float>();
    float* out = result.data_ptr<float>();
    int64_t n = x.num_elements();

    for (int64_t i = 0; i < n; ++i) {
        out[i] = std::sqrt(in[i]);
    }

    return {std::move(result)};
}

std::vector<Tensor> cpu_exp(const std::vector<const Tensor*>& inputs,
                            const OpContext& /*ctx*/) {
    validate_input_count(inputs, 1, "Exp");  // LOW-01 fix
    const Tensor& x = *inputs[0];
    Tensor result(x.shape(), x.dtype());

    const float* in = x.data_ptr<float>();
    float* out = result.data_ptr<float>();
    int64_t n = x.num_elements();

    for (int64_t i = 0; i < n; ++i) {
        out[i] = std::exp(in[i]);
    }

    return {std::move(result)};
}

std::vector<Tensor> cpu_log(const std::vector<const Tensor*>& inputs,
                            const OpContext& /*ctx*/) {
    validate_input_count(inputs, 1, "Log");  // LOW-01 fix
    const Tensor& x = *inputs[0];
    Tensor result(x.shape(), x.dtype());

    const float* in = x.data_ptr<float>();
    float* out = result.data_ptr<float>();
    int64_t n = x.num_elements();

    for (int64_t i = 0; i < n; ++i) {
        out[i] = std::log(in[i]);
    }

    return {std::move(result)};
}

std::vector<Tensor> cpu_pow(const std::vector<const Tensor*>& inputs,
                            const OpContext& ctx) {
    return binary_elementwise(inputs, ctx,
        [](float a, float b) { return std::pow(a, b); }, "Pow");
}

std::vector<Tensor> cpu_neg(const std::vector<const Tensor*>& inputs,
                            const OpContext& /*ctx*/) {
    validate_input_count(inputs, 1, "Neg");  // LOW-01 fix
    const Tensor& x = *inputs[0];
    Tensor result(x.shape(), x.dtype());

    const float* in = x.data_ptr<float>();
    float* out = result.data_ptr<float>();
    int64_t n = x.num_elements();

    for (int64_t i = 0; i < n; ++i) {
        out[i] = -in[i];
    }

    return {std::move(result)};
}

std::vector<Tensor> cpu_abs(const std::vector<const Tensor*>& inputs,
                            const OpContext& /*ctx*/) {
    validate_input_count(inputs, 1, "Abs");  // LOW-01 fix
    const Tensor& x = *inputs[0];
    Tensor result(x.shape(), x.dtype());

    const float* in = x.data_ptr<float>();
    float* out = result.data_ptr<float>();
    int64_t n = x.num_elements();

    for (int64_t i = 0; i < n; ++i) {
        out[i] = std::abs(in[i]);
    }

    return {std::move(result)};
}

// Register operators at static initialization
struct MathOpsRegistrar {
    MathOpsRegistrar() {
        auto& reg = OperatorRegistry::instance();
        reg.register_op("Add", cpu_add);
        reg.register_op("Sub", cpu_sub);
        reg.register_op("Mul", cpu_mul);
        reg.register_op("Div", cpu_div);
        reg.register_op("MatMul", cpu_matmul);
        reg.register_op("Gemm", cpu_gemm);
        reg.register_op("Sqrt", cpu_sqrt);
        reg.register_op("Exp", cpu_exp);
        reg.register_op("Log", cpu_log);
        reg.register_op("Pow", cpu_pow);
        reg.register_op("Neg", cpu_neg);
        reg.register_op("Abs", cpu_abs);
    }
};

static MathOpsRegistrar math_ops_registrar;

} // anonymous namespace
} // namespace ops
} // namespace pyflame_rt
