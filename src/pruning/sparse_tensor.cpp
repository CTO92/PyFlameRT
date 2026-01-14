#include "pyflame_rt/pruning/sparse_tensor.hpp"
#include "pyflame_rt/types.hpp"  // For checked_multiply, checked_product
#include <algorithm>
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <limits>

namespace pyflame_rt {
namespace pruning {

// ============================================================================
// Security Helper Functions
// ============================================================================

namespace {

/// Validate that indices are within bounds for 2D COO tensor
inline void validate_coo_indices(const std::vector<int64_t>& indices,
                                  size_t nnz,
                                  int64_t rows, int64_t cols) {
    for (size_t i = 0; i < nnz; ++i) {
        int64_t row = indices[i * 2];
        int64_t col = indices[i * 2 + 1];
        if (row < 0 || row >= rows) {
            throw std::out_of_range(
                "COO row index " + std::to_string(row) + " out of range [0, " +
                std::to_string(rows) + ")");
        }
        if (col < 0 || col >= cols) {
            throw std::out_of_range(
                "COO column index " + std::to_string(col) + " out of range [0, " +
                std::to_string(cols) + ")");
        }
    }
}

/// Validate CSR index bounds
inline void validate_csr_indices(const std::vector<int64_t>& indices,
                                  const std::vector<int64_t>& indptr,
                                  int64_t cols) {
    for (size_t j = 0; j < indices.size(); ++j) {
        if (indices[j] < 0 || indices[j] >= cols) {
            throw std::out_of_range(
                "CSR column index " + std::to_string(indices[j]) + " out of range [0, " +
                std::to_string(cols) + ")");
        }
    }
}

/// Safe index calculation with overflow check
inline size_t safe_index_2d(int64_t row, int64_t col, int64_t cols) {
    // Both row and col should already be bounds-checked
    if (row < 0 || col < 0 || cols <= 0) {
        throw std::invalid_argument("Invalid index or dimension");
    }
    // Check for overflow
    if (row > std::numeric_limits<int64_t>::max() / cols) {
        throw std::overflow_error("Index calculation overflow");
    }
    int64_t idx = row * cols + col;
    if (idx < 0) {
        throw std::overflow_error("Index calculation overflow");
    }
    return static_cast<size_t>(idx);
}

} // anonymous namespace

// ============================================================================
// SparseTensor Implementation
// ============================================================================

SparseTensor::SparseTensor(const Tensor& dense, float threshold) {
    shape_ = dense.shape();
    dtype_ = dense.dtype();
    format_ = SparseFormat::COO;

    const float* data = static_cast<const float*>(dense.data());
    size_t total = dense.num_elements();

    // Count and collect non-zero elements
    for (size_t i = 0; i < total; ++i) {
        if (std::abs(data[i]) > threshold) {
            values_.push_back(data[i]);
            // Store flat index for COO
            indices_.push_back(static_cast<int64_t>(i));
        }
    }

    nnz_ = values_.size();
}

SparseTensor SparseTensor::from_masked(const Tensor& dense, const Tensor& mask) {
    if (dense.num_elements() != mask.num_elements()) {
        throw std::invalid_argument("Dense tensor and mask must have same number of elements");
    }

    SparseTensor sparse;
    sparse.shape_ = dense.shape();
    sparse.dtype_ = dense.dtype();
    sparse.format_ = SparseFormat::COO;

    const float* data = static_cast<const float*>(dense.data());
    const float* mask_data = static_cast<const float*>(mask.data());
    size_t total = dense.num_elements();

    for (size_t i = 0; i < total; ++i) {
        if (mask_data[i] != 0.0f) {
            sparse.values_.push_back(data[i]);
            sparse.indices_.push_back(static_cast<int64_t>(i));
        }
    }

    sparse.nnz_ = sparse.values_.size();
    return sparse;
}

SparseTensor SparseTensor::from_coo(
    const std::vector<int64_t>& shape,
    const std::vector<int64_t>& row_indices,
    const std::vector<int64_t>& col_indices,
    const std::vector<float>& values,
    DType dtype)
{
    if (row_indices.size() != col_indices.size() ||
        row_indices.size() != values.size()) {
        throw std::invalid_argument("COO arrays must have same length");
    }
    if (shape.size() != 2) {
        throw std::invalid_argument("COO format currently supports 2D tensors only");
    }

    SparseTensor sparse;
    sparse.shape_ = shape;
    sparse.dtype_ = dtype;
    sparse.format_ = SparseFormat::COO;
    sparse.values_ = values;
    sparse.nnz_ = values.size();

    // Security: validate indices are within bounds (HIGH-P1 fix)
    for (size_t i = 0; i < row_indices.size(); ++i) {
        if (row_indices[i] < 0 || row_indices[i] >= shape[0]) {
            throw std::out_of_range(
                "Row index " + std::to_string(row_indices[i]) +
                " out of range [0, " + std::to_string(shape[0]) + ")");
        }
        if (col_indices[i] < 0 || col_indices[i] >= shape[1]) {
            throw std::out_of_range(
                "Column index " + std::to_string(col_indices[i]) +
                " out of range [0, " + std::to_string(shape[1]) + ")");
        }
    }

    // Security: check for overflow in indices resize (MED-P3 fix)
    if (row_indices.size() > std::numeric_limits<size_t>::max() / 2) {
        throw std::overflow_error("Row indices size too large for interleaving");
    }

    // Interleave row and column indices
    sparse.indices_.resize(row_indices.size() * 2);
    for (size_t i = 0; i < row_indices.size(); ++i) {
        sparse.indices_[i * 2] = row_indices[i];
        sparse.indices_[i * 2 + 1] = col_indices[i];
    }

    return sparse;
}

SparseTensor SparseTensor::from_csr(
    const std::vector<int64_t>& shape,
    const std::vector<int64_t>& indptr,
    const std::vector<int64_t>& indices,
    const std::vector<float>& values,
    DType dtype)
{
    if (indices.size() != values.size()) {
        throw std::invalid_argument("CSR indices and values must have same length");
    }
    if (shape.size() != 2) {
        throw std::invalid_argument("CSR format currently supports 2D tensors only");
    }
    if (static_cast<int64_t>(indptr.size()) != shape[0] + 1) {
        throw std::invalid_argument("CSR indptr must have length rows + 1");
    }

    SparseTensor sparse;
    sparse.shape_ = shape;
    sparse.dtype_ = dtype;
    sparse.format_ = SparseFormat::CSR;
    sparse.values_ = values;
    sparse.indices_ = indices;
    sparse.indptr_ = indptr;
    sparse.nnz_ = values.size();

    return sparse;
}

Tensor SparseTensor::to_dense() const {
    Tensor dense(shape_, dtype_);
    float* data = static_cast<float*>(dense.data());
    size_t total_elements = num_elements();
    std::fill(data, data + total_elements, 0.0f);

    if (format_ == SparseFormat::COO) {
        if (shape_.size() == 2 && indices_.size() == nnz_ * 2) {
            // 2D COO with interleaved indices
            int64_t rows = shape_[0];
            int64_t cols = shape_[1];

            // Security: validate indices before accessing (HIGH-P1 fix)
            for (size_t i = 0; i < nnz_; ++i) {
                int64_t row = indices_[i * 2];
                int64_t col = indices_[i * 2 + 1];

                // Bounds check
                if (row < 0 || row >= rows) {
                    throw std::out_of_range(
                        "COO row index " + std::to_string(row) + " out of range");
                }
                if (col < 0 || col >= cols) {
                    throw std::out_of_range(
                        "COO column index " + std::to_string(col) + " out of range");
                }

                // Safe index calculation (HIGH-P2 fix)
                size_t idx = safe_index_2d(row, col, cols);
                data[idx] = values_[i];
            }
        } else {
            // Flat indices
            for (size_t i = 0; i < nnz_; ++i) {
                int64_t flat_idx = indices_[i];
                if (flat_idx < 0 || static_cast<size_t>(flat_idx) >= total_elements) {
                    throw std::out_of_range(
                        "COO flat index " + std::to_string(flat_idx) + " out of range");
                }
                data[flat_idx] = values_[i];
            }
        }
    } else if (format_ == SparseFormat::CSR) {
        int64_t rows = shape_[0];
        int64_t cols = shape_[1];

        for (size_t row = 0; row < static_cast<size_t>(rows); ++row) {
            // Security: validate indptr bounds
            if (row + 1 >= indptr_.size()) {
                throw std::out_of_range("CSR indptr index out of range");
            }

            for (int64_t j = indptr_[row]; j < indptr_[row + 1]; ++j) {
                // Security: validate j is in range
                if (j < 0 || static_cast<size_t>(j) >= indices_.size()) {
                    throw std::out_of_range("CSR indices index out of range");
                }

                int64_t col = indices_[j];
                if (col < 0 || col >= cols) {
                    throw std::out_of_range(
                        "CSR column index " + std::to_string(col) + " out of range");
                }

                // Safe index calculation (HIGH-P2 fix)
                size_t idx = safe_index_2d(static_cast<int64_t>(row), col, cols);
                data[idx] = values_[j];
            }
        }
    } else if (format_ == SparseFormat::CSC) {
        int64_t rows = shape_[0];
        int64_t cols = shape_[1];

        for (size_t col = 0; col < static_cast<size_t>(cols); ++col) {
            // Security: validate indptr bounds
            if (col + 1 >= indptr_.size()) {
                throw std::out_of_range("CSC indptr index out of range");
            }

            for (int64_t j = indptr_[col]; j < indptr_[col + 1]; ++j) {
                // Security: validate j is in range
                if (j < 0 || static_cast<size_t>(j) >= indices_.size()) {
                    throw std::out_of_range("CSC indices index out of range");
                }

                int64_t row = indices_[j];
                if (row < 0 || row >= rows) {
                    throw std::out_of_range(
                        "CSC row index " + std::to_string(row) + " out of range");
                }

                // Safe index calculation (HIGH-P2 fix)
                size_t idx = safe_index_2d(row, static_cast<int64_t>(col), cols);
                data[idx] = values_[j];
            }
        }
    }

    return dense;
}

SparseTensor SparseTensor::to_format(SparseFormat new_format) const {
    if (format_ == new_format) {
        return *this;
    }

    SparseTensor result;
    result.shape_ = shape_;
    result.dtype_ = dtype_;
    result.format_ = new_format;

    if (shape_.size() != 2) {
        throw std::runtime_error("Format conversion only supported for 2D tensors");
    }

    // Convert via dense (simple but not optimal)
    Tensor dense = to_dense();

    if (new_format == SparseFormat::CSR) {
        result = to_sparse(dense, SparseFormat::CSR);
    } else if (new_format == SparseFormat::CSC) {
        result = to_sparse(dense, SparseFormat::CSC);
    } else {
        result = to_sparse(dense, SparseFormat::COO);
    }

    return result;
}

float SparseTensor::sparsity() const {
    size_t total = num_elements();
    if (total == 0) return 0.0f;
    return 1.0f - static_cast<float>(nnz_) / total;
}

size_t SparseTensor::num_elements() const {
    if (shape_.empty()) return 0;

    // Security: use overflow-safe multiplication (CRIT-P1 fix)
    size_t total = 1;
    for (auto dim : shape_) {
        if (dim < 0) {
            throw std::invalid_argument("Negative dimension in shape");
        }
        size_t dim_size = static_cast<size_t>(dim);
        // Check for overflow before multiplication
        if (dim_size > 0 && total > std::numeric_limits<size_t>::max() / dim_size) {
            throw std::overflow_error("Shape dimensions cause integer overflow");
        }
        total *= dim_size;
    }
    return total;
}

std::vector<int64_t> SparseTensor::row_indices() const {
    if (format_ != SparseFormat::COO || shape_.size() != 2) {
        throw std::runtime_error("row_indices only available for 2D COO format");
    }

    std::vector<int64_t> rows(nnz_);
    if (indices_.size() == nnz_ * 2) {
        for (size_t i = 0; i < nnz_; ++i) {
            rows[i] = indices_[i * 2];
        }
    } else {
        // Flat indices - compute row from flat index
        int64_t cols = shape_[1];
        for (size_t i = 0; i < nnz_; ++i) {
            rows[i] = indices_[i] / cols;
        }
    }
    return rows;
}

std::vector<int64_t> SparseTensor::col_indices() const {
    if (format_ != SparseFormat::COO || shape_.size() != 2) {
        throw std::runtime_error("col_indices only available for 2D COO format");
    }

    std::vector<int64_t> cols(nnz_);
    if (indices_.size() == nnz_ * 2) {
        for (size_t i = 0; i < nnz_; ++i) {
            cols[i] = indices_[i * 2 + 1];
        }
    } else {
        int64_t num_cols = shape_[1];
        for (size_t i = 0; i < nnz_; ++i) {
            cols[i] = indices_[i] % num_cols;
        }
    }
    return cols;
}

size_t SparseTensor::memory_bytes() const {
    size_t bytes = values_.size() * sizeof(float);
    bytes += indices_.size() * sizeof(int64_t);
    bytes += indptr_.size() * sizeof(int64_t);
    return bytes;
}

float SparseTensor::compression_ratio() const {
    size_t dense_bytes = num_elements() * sizeof(float);
    size_t sparse_bytes = memory_bytes();
    if (sparse_bytes == 0) return 1.0f;
    return static_cast<float>(dense_bytes) / sparse_bytes;
}

bool SparseTensor::is_valid() const {
    if (nnz_ != values_.size()) return false;

    if (format_ == SparseFormat::COO) {
        // Check indices are within bounds
        if (shape_.size() == 2 && indices_.size() == nnz_ * 2) {
            for (size_t i = 0; i < nnz_; ++i) {
                if (indices_[i * 2] < 0 || indices_[i * 2] >= shape_[0]) return false;
                if (indices_[i * 2 + 1] < 0 || indices_[i * 2 + 1] >= shape_[1]) return false;
            }
        }
    } else if (format_ == SparseFormat::CSR) {
        if (shape_.size() != 2) return false;
        if (indptr_.size() != static_cast<size_t>(shape_[0] + 1)) return false;
        if (indices_.size() != nnz_) return false;
    }

    return true;
}

SparseTensor SparseTensor::transpose() const {
    if (shape_.size() != 2) {
        throw std::runtime_error("Transpose only supported for 2D tensors");
    }

    SparseTensor result;
    result.shape_ = {shape_[1], shape_[0]};
    result.dtype_ = dtype_;
    result.nnz_ = nnz_;
    result.values_ = values_;

    if (format_ == SparseFormat::CSR) {
        // CSR transpose = CSC
        result.format_ = SparseFormat::CSC;
        result.indptr_ = indptr_;
        result.indices_ = indices_;
    } else if (format_ == SparseFormat::CSC) {
        result.format_ = SparseFormat::CSR;
        result.indptr_ = indptr_;
        result.indices_ = indices_;
    } else {
        // COO: swap row and column indices
        result.format_ = SparseFormat::COO;
        if (indices_.size() == nnz_ * 2) {
            result.indices_.resize(nnz_ * 2);
            for (size_t i = 0; i < nnz_; ++i) {
                result.indices_[i * 2] = indices_[i * 2 + 1];
                result.indices_[i * 2 + 1] = indices_[i * 2];
            }
        }
    }

    return result;
}

Tensor SparseTensor::matvec(const Tensor& vec) const {
    return sparse_matvec(*this, vec);
}

Tensor SparseTensor::matmul(const Tensor& dense) const {
    return sparse_matmul(*this, dense);
}

// ============================================================================
// Free Functions
// ============================================================================

Tensor sparse_matmul(const SparseTensor& a, const Tensor& b) {
    if (a.shape().size() != 2 || b.shape().size() < 1) {
        throw std::invalid_argument("Invalid dimensions for sparse matmul");
    }

    int64_t m = a.shape()[0];
    int64_t k = a.shape()[1];
    int64_t n = (b.shape().size() == 1) ? 1 : b.shape()[1];

    if (k != b.shape()[0]) {
        throw std::invalid_argument("Dimension mismatch in sparse matmul");
    }

    std::vector<int64_t> out_shape = {m, n};
    if (b.shape().size() == 1) {
        out_shape = {m};
    }

    Tensor result(out_shape, a.dtype());
    float* out = static_cast<float*>(result.data());
    std::fill(out, out + result.num_elements(), 0.0f);

    const float* b_data = static_cast<const float*>(b.data());
    const auto& values = a.values();
    const auto& indices = a.indices();
    const auto& indptr = a.indptr();

    if (a.format() == SparseFormat::CSR) {
        // CSR SpMM
        for (int64_t row = 0; row < m; ++row) {
            for (int64_t j = indptr[row]; j < indptr[row + 1]; ++j) {
                int64_t col = indices[j];
                float val = values[j];
                for (int64_t c = 0; c < n; ++c) {
                    out[row * n + c] += val * b_data[col * n + c];
                }
            }
        }
    } else {
        // Convert to dense and multiply (fallback)
        Tensor dense_a = a.to_dense();
        // Would call dense matmul here
        throw std::runtime_error("Sparse matmul only implemented for CSR format");
    }

    return result;
}

Tensor sparse_matvec(const SparseTensor& a, const Tensor& x) {
    if (a.shape().size() != 2 || x.shape().size() != 1) {
        throw std::invalid_argument("Invalid dimensions for sparse matvec");
    }

    int64_t m = a.shape()[0];
    int64_t k = a.shape()[1];

    if (k != x.shape()[0]) {
        throw std::invalid_argument("Dimension mismatch in sparse matvec");
    }

    Tensor result({m}, a.dtype());
    float* out = static_cast<float*>(result.data());
    std::fill(out, out + m, 0.0f);

    const float* x_data = static_cast<const float*>(x.data());
    const auto& values = a.values();
    const auto& indices = a.indices();
    const auto& indptr = a.indptr();

    if (a.format() == SparseFormat::CSR) {
        for (int64_t row = 0; row < m; ++row) {
            for (int64_t j = indptr[row]; j < indptr[row + 1]; ++j) {
                int64_t col = indices[j];
                out[row] += values[j] * x_data[col];
            }
        }
    } else {
        throw std::runtime_error("Sparse matvec only implemented for CSR format");
    }

    return result;
}

SparseTensor to_sparse(const Tensor& dense, SparseFormat format) {
    return to_sparse(dense, 0.0f, format);
}

SparseTensor to_sparse(const Tensor& dense, float threshold, SparseFormat format) {
    if (dense.shape().size() != 2) {
        // For non-2D, use COO with flat indices
        return SparseTensor(dense, threshold);
    }

    int64_t rows = dense.shape()[0];
    int64_t cols = dense.shape()[1];
    const float* data = static_cast<const float*>(dense.data());

    if (format == SparseFormat::CSR) {
        std::vector<int64_t> indptr;
        std::vector<int64_t> indices;
        std::vector<float> values;

        indptr.push_back(0);
        for (int64_t r = 0; r < rows; ++r) {
            for (int64_t c = 0; c < cols; ++c) {
                float val = data[r * cols + c];
                if (std::abs(val) > threshold) {
                    indices.push_back(c);
                    values.push_back(val);
                }
            }
            indptr.push_back(static_cast<int64_t>(indices.size()));
        }

        return SparseTensor::from_csr(dense.shape(), indptr, indices, values, dense.dtype());

    } else if (format == SparseFormat::COO) {
        std::vector<int64_t> row_indices;
        std::vector<int64_t> col_indices;
        std::vector<float> values;

        for (int64_t r = 0; r < rows; ++r) {
            for (int64_t c = 0; c < cols; ++c) {
                float val = data[r * cols + c];
                if (std::abs(val) > threshold) {
                    row_indices.push_back(r);
                    col_indices.push_back(c);
                    values.push_back(val);
                }
            }
        }

        return SparseTensor::from_coo(dense.shape(), row_indices, col_indices, values, dense.dtype());
    }

    // Default to COO
    return SparseTensor(dense, threshold);
}

Tensor to_dense(const SparseTensor& sparse) {
    return sparse.to_dense();
}

Tensor apply_mask(const Tensor& tensor, const Tensor& mask) {
    if (tensor.num_elements() != mask.num_elements()) {
        throw std::invalid_argument("Tensor and mask must have same number of elements");
    }

    Tensor result(tensor.shape(), tensor.dtype());
    const float* src = static_cast<const float*>(tensor.data());
    const float* mask_data = static_cast<const float*>(mask.data());
    float* dst = static_cast<float*>(result.data());

    for (size_t i = 0; i < tensor.num_elements(); ++i) {
        dst[i] = src[i] * mask_data[i];
    }

    return result;
}

size_t count_nonzero(const Tensor& tensor) {
    size_t count = 0;
    const float* data = static_cast<const float*>(tensor.data());
    for (size_t i = 0; i < tensor.num_elements(); ++i) {
        if (data[i] != 0.0f) count++;
    }
    return count;
}

float compute_sparsity(const Tensor& tensor) {
    size_t total = tensor.num_elements();
    if (total == 0) return 0.0f;
    size_t nonzero = count_nonzero(tensor);
    return 1.0f - static_cast<float>(nonzero) / total;
}

} // namespace pruning
} // namespace pyflame_rt
