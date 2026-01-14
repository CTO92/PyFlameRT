#include "pyflame_rt/tensor.hpp"
#include "pyflame_rt/types.hpp"
#include <cstdlib>
#include <algorithm>
#include <stdexcept>
#include <limits>

#ifdef _WIN32
#include <malloc.h>
#define aligned_alloc(alignment, size) _aligned_malloc(size, alignment)
#define aligned_free(ptr) _aligned_free(ptr)
#else
#define aligned_free(ptr) std::free(ptr)
#endif

namespace pyflame_rt {

namespace {
/// Validate shape dimensions - reject negative values (CRIT-05 fix)
void validate_shape(const std::vector<int64_t>& shape) {
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] < 0) {
            throw std::invalid_argument(
                "Negative dimension at axis " + std::to_string(i) +
                ": " + std::to_string(shape[i]));
        }
    }
}
} // anonymous namespace

Tensor::Tensor(const std::vector<int64_t>& shape, DType dtype)
    : shape_(shape)
    , dtype_(dtype)
{
    // Security: validate shape before computing size (CRIT-05)
    validate_shape(shape_);

    size_t bytes = size_bytes();  // Uses checked arithmetic now
    if (bytes > 0) {
        // Allocate aligned memory (64-byte alignment for SIMD)
        void* ptr = aligned_alloc(64, bytes);
        if (!ptr) {
            throw std::bad_alloc();
        }
        owned_data_ = std::shared_ptr<void>(ptr, [](void* p) { aligned_free(p); });
        data_ = owned_data_.get();
        std::memset(data_, 0, bytes);
    }
}

Tensor::Tensor(void* data, const std::vector<int64_t>& shape, DType dtype)
    : data_(data)
    , shape_(shape)
    , dtype_(dtype)
    , owned_data_(nullptr)  // Non-owning
{
}

Tensor::Tensor(const Tensor& other)
    : shape_(other.shape_)
    , dtype_(other.dtype_)
{
    if (other.is_valid()) {
        size_t bytes = size_bytes();
        void* ptr = aligned_alloc(64, bytes);
        if (!ptr) {
            throw std::bad_alloc();
        }
        owned_data_ = std::shared_ptr<void>(ptr, [](void* p) { aligned_free(p); });
        data_ = owned_data_.get();
        std::memcpy(data_, other.data_, bytes);
    }
}

Tensor::Tensor(Tensor&& other) noexcept
    : data_(other.data_)
    , shape_(std::move(other.shape_))
    , dtype_(other.dtype_)
    , owned_data_(std::move(other.owned_data_))
{
    other.data_ = nullptr;
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        Tensor tmp(other);
        std::swap(data_, tmp.data_);
        std::swap(shape_, tmp.shape_);
        std::swap(dtype_, tmp.dtype_);
        std::swap(owned_data_, tmp.owned_data_);
    }
    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        data_ = other.data_;
        shape_ = std::move(other.shape_);
        dtype_ = other.dtype_;
        owned_data_ = std::move(other.owned_data_);
        other.data_ = nullptr;
    }
    return *this;
}

/// Calculate total elements with overflow protection (CRIT-05 fix)
int64_t Tensor::num_elements() const {
    if (shape_.empty()) return 0;
    // Use checked_product for overflow-safe multiplication
    return checked_product(shape_);
}

/// Calculate size in bytes with overflow protection (CRIT-05 fix)
size_t Tensor::size_bytes() const {
    int64_t elements = num_elements();
    size_t elem_size = dtype_size(dtype_);
    // Check for overflow when multiplying by element size
    if (elements > 0 && static_cast<size_t>(elements) > std::numeric_limits<size_t>::max() / elem_size) {
        throw std::overflow_error("Tensor size in bytes would overflow");
    }
    return static_cast<size_t>(elements) * elem_size;
}

Tensor Tensor::clone() const {
    return Tensor(*this);  // Uses copy constructor
}

Tensor Tensor::view() const {
    return Tensor(const_cast<void*>(data_), shape_, dtype_);
}

void Tensor::zero() {
    if (is_valid()) {
        std::memset(data_, 0, size_bytes());
    }
}

Tensor Tensor::reshape(const std::vector<int64_t>& new_shape) const {
    // Security: validate new shape (MED-03 fix)
    validate_shape(new_shape);

    // Calculate new total elements with overflow protection
    int64_t new_total = checked_product(new_shape);

    if (new_total != num_elements()) {
        throw std::invalid_argument(
            "Cannot reshape tensor: element count mismatch (" +
            std::to_string(new_total) + " vs " + std::to_string(num_elements()) + ")");
    }

    Tensor result = clone();
    result.shape_ = new_shape;
    return result;
}

} // namespace pyflame_rt
