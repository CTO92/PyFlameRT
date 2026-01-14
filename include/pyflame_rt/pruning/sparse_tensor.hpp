#pragma once

#include "pyflame_rt/tensor.hpp"
#include <vector>
#include <memory>

namespace pyflame_rt {
namespace pruning {

/// Sparse tensor format
enum class SparseFormat {
    COO,    // Coordinate format
    CSR,    // Compressed sparse row
    CSC,    // Compressed sparse column
    BSR     // Block sparse row
};

/// Sparse tensor representation
class SparseTensor {
public:
    SparseTensor() = default;

    /// Create from dense tensor with threshold
    explicit SparseTensor(const Tensor& dense, float threshold = 0.0f);

    /// Create from dense tensor with mask
    static SparseTensor from_masked(const Tensor& dense, const Tensor& mask);

    /// Create from COO data
    static SparseTensor from_coo(
        const std::vector<int64_t>& shape,
        const std::vector<int64_t>& row_indices,
        const std::vector<int64_t>& col_indices,
        const std::vector<float>& values,
        DType dtype = DType::Float32);

    /// Create from CSR data
    static SparseTensor from_csr(
        const std::vector<int64_t>& shape,
        const std::vector<int64_t>& indptr,
        const std::vector<int64_t>& indices,
        const std::vector<float>& values,
        DType dtype = DType::Float32);

    /// Convert to dense tensor
    Tensor to_dense() const;

    /// Convert to different sparse format
    SparseTensor to_format(SparseFormat format) const;

    /// Get non-zero count
    size_t nnz() const { return nnz_; }

    /// Get sparsity ratio
    float sparsity() const;

    /// Get density (1 - sparsity)
    float density() const { return 1.0f - sparsity(); }

    /// Get shape
    const std::vector<int64_t>& shape() const { return shape_; }

    /// Get number of dimensions
    size_t ndim() const { return shape_.size(); }

    /// Get total number of elements (if dense)
    size_t num_elements() const;

    /// Get format
    SparseFormat format() const { return format_; }

    /// Get data type
    DType dtype() const { return dtype_; }

    /// Get values array
    const std::vector<float>& values() const { return values_; }

    /// Get indices (for COO: interleaved row/col, for CSR/CSC: column/row indices)
    const std::vector<int64_t>& indices() const { return indices_; }

    /// Get indptr (for CSR/CSC)
    const std::vector<int64_t>& indptr() const { return indptr_; }

    /// Get row indices (COO only)
    std::vector<int64_t> row_indices() const;

    /// Get column indices (COO only)
    std::vector<int64_t> col_indices() const;

    /// Memory size in bytes
    size_t memory_bytes() const;

    /// Compression ratio vs dense
    float compression_ratio() const;

    /// Check if empty
    bool empty() const { return nnz_ == 0; }

    /// Check if valid
    bool is_valid() const;

    // ========================================================================
    // Operations
    // ========================================================================

    /// Transpose
    SparseTensor transpose() const;

    /// Element-wise addition with dense tensor
    Tensor add(const Tensor& dense) const;

    /// Sparse matrix-vector multiplication
    Tensor matvec(const Tensor& vec) const;

    /// Sparse matrix-matrix multiplication
    Tensor matmul(const Tensor& dense) const;

private:
    std::vector<int64_t> shape_;
    SparseFormat format_ = SparseFormat::COO;
    size_t nnz_ = 0;
    DType dtype_ = DType::Float32;

    std::vector<float> values_;
    std::vector<int64_t> indices_;
    std::vector<int64_t> indptr_;

    void convert_coo_to_csr();
    void convert_csr_to_coo();
    void convert_coo_to_csc();
    void convert_csc_to_coo();
};

/// Sparse matrix multiplication
Tensor sparse_matmul(const SparseTensor& a, const Tensor& b);

/// Sparse matrix-vector multiplication
Tensor sparse_matvec(const SparseTensor& a, const Tensor& x);

/// Convert dense tensor to sparse
SparseTensor to_sparse(const Tensor& dense, SparseFormat format = SparseFormat::CSR);

/// Convert dense tensor to sparse with threshold
SparseTensor to_sparse(const Tensor& dense, float threshold, SparseFormat format = SparseFormat::CSR);

/// Convert sparse tensor to dense
Tensor to_dense(const SparseTensor& sparse);

/// Apply sparsity mask to tensor
Tensor apply_mask(const Tensor& tensor, const Tensor& mask);

/// Count non-zero elements in tensor
size_t count_nonzero(const Tensor& tensor);

/// Compute sparsity of tensor
float compute_sparsity(const Tensor& tensor);

} // namespace pruning
} // namespace pyflame_rt
