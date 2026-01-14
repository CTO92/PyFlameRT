#pragma once

#include "pyflame_rt/types.hpp"
#include <memory>
#include <vector>
#include <cstring>
#include <functional>

namespace pyflame_rt {

/// Tensor data container with optional ownership
class Tensor {
public:
    /// Create empty tensor
    Tensor() = default;

    /// Create tensor with owned memory
    Tensor(const std::vector<int64_t>& shape, DType dtype);

    /// Create tensor with borrowed memory (non-owning view)
    Tensor(void* data, const std::vector<int64_t>& shape, DType dtype);

    /// Copy constructor
    Tensor(const Tensor& other);

    /// Move constructor
    Tensor(Tensor&& other) noexcept;

    /// Copy assignment
    Tensor& operator=(const Tensor& other);

    /// Move assignment
    Tensor& operator=(Tensor&& other) noexcept;

    ~Tensor() = default;

    // Accessors
    const std::vector<int64_t>& shape() const { return shape_; }
    DType dtype() const { return dtype_; }
    size_t ndim() const { return shape_.size(); }
    int64_t size(size_t dim) const {
        if (dim >= shape_.size()) {
            throw std::out_of_range("Dimension index " + std::to_string(dim) +
                " out of range for tensor with " + std::to_string(shape_.size()) + " dimensions");
        }
        return shape_[dim];
    }

    int64_t num_elements() const;
    size_t size_bytes() const;

    /// Raw data pointer
    void* data() { return data_; }
    const void* data() const { return data_; }

    /// Typed data access
    template<typename T>
    T* data_ptr() { return static_cast<T*>(data_); }

    template<typename T>
    const T* data_ptr() const { return static_cast<const T*>(data_); }

    /// Check if tensor owns its memory
    bool owns_data() const { return owned_data_ != nullptr; }

    /// Create a deep copy
    Tensor clone() const;

    /// Create a view (non-owning)
    Tensor view() const;

    /// Check if tensor is valid
    bool is_valid() const { return data_ != nullptr; }

    /// Fill tensor with value
    template<typename T>
    void fill(T value);

    /// Zero out the tensor
    void zero();

    /// Reshape tensor (must have same total elements)
    Tensor reshape(const std::vector<int64_t>& new_shape) const;

private:
    void* data_ = nullptr;
    std::vector<int64_t> shape_;
    DType dtype_ = DType::Float32;
    std::shared_ptr<void> owned_data_;  // For memory management
};

// Template implementation
template<typename T>
void Tensor::fill(T value) {
    // Security fix CRIT-01: Validate type size matches tensor dtype
    if (sizeof(T) != dtype_size(dtype_)) {
        throw std::invalid_argument(
            "Type size mismatch in fill(): sizeof(T)=" + std::to_string(sizeof(T)) +
            " but dtype size=" + std::to_string(dtype_size(dtype_)) +
            ". Use the correct type for dtype " + dtype_name(dtype_));
    }
    if (!is_valid()) {
        throw std::runtime_error("Cannot fill invalid tensor");
    }
    T* ptr = data_ptr<T>();
    int64_t n = num_elements();
    for (int64_t i = 0; i < n; ++i) {
        ptr[i] = value;
    }
}

} // namespace pyflame_rt
