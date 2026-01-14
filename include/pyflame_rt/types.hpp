#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <limits>

namespace pyflame_rt {

// ============================================================================
// Security Helper Functions
// ============================================================================

/// Overflow-safe multiplication for size calculations
/// Throws std::overflow_error if overflow would occur
inline int64_t checked_multiply(int64_t a, int64_t b) {
    if (a == 0 || b == 0) return 0;
    if (a < 0 || b < 0) {
        throw std::invalid_argument("Negative dimensions not allowed in size calculation");
    }
    if (a > std::numeric_limits<int64_t>::max() / b) {
        throw std::overflow_error("Integer overflow in size calculation");
    }
    return a * b;
}

/// Overflow-safe product of multiple values
inline int64_t checked_product(const std::vector<int64_t>& values) {
    int64_t result = 1;
    for (int64_t v : values) {
        result = checked_multiply(result, v);
    }
    return result;
}

/// Check if adding two size_t values would overflow
inline size_t checked_add(size_t a, size_t b) {
    if (a > std::numeric_limits<size_t>::max() - b) {
        throw std::overflow_error("Integer overflow in addition");
    }
    return a + b;
}

/// Supported tensor data types
enum class DType : uint8_t {
    Float32 = 0,
    Float16 = 1,
    BFloat16 = 2,
    Float64 = 3,
    Int64 = 4,
    Int32 = 5,
    Int16 = 6,
    Int8 = 7,
    UInt8 = 8,
    Bool = 9
};

/// Check if a raw integer value is a valid DType enum value
/// CRIT-04 fix: Use actual enum value instead of hardcoded constant
inline bool validate_dtype_value(uint32_t value) {
    // DType::Bool is the last valid enum value
    // Using static_cast ensures this stays in sync with enum definition
    return value <= static_cast<uint32_t>(DType::Bool);
}

/// Get size in bytes for a data type
inline size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::Float32: return 4;
        case DType::Float16: return 2;
        case DType::BFloat16: return 2;
        case DType::Float64: return 8;
        case DType::Int64: return 8;
        case DType::Int32: return 4;
        case DType::Int16: return 2;
        case DType::Int8: return 1;
        case DType::UInt8: return 1;
        case DType::Bool: return 1;
        default: throw std::invalid_argument("Unknown dtype");
    }
}

/// Convert dtype to string name
inline std::string dtype_name(DType dtype) {
    switch (dtype) {
        case DType::Float32: return "float32";
        case DType::Float16: return "float16";
        case DType::BFloat16: return "bfloat16";
        case DType::Float64: return "float64";
        case DType::Int64: return "int64";
        case DType::Int32: return "int32";
        case DType::Int16: return "int16";
        case DType::Int8: return "int8";
        case DType::UInt8: return "uint8";
        case DType::Bool: return "bool";
        default: throw std::invalid_argument("Unknown dtype");
    }
}

/// Parse dtype from string name
inline DType dtype_from_name(const std::string& name) {
    if (name == "float32") return DType::Float32;
    if (name == "float16") return DType::Float16;
    if (name == "bfloat16") return DType::BFloat16;
    if (name == "float64") return DType::Float64;
    if (name == "int64") return DType::Int64;
    if (name == "int32") return DType::Int32;
    if (name == "int16") return DType::Int16;
    if (name == "int8") return DType::Int8;
    if (name == "uint8") return DType::UInt8;
    if (name == "bool") return DType::Bool;
    throw std::invalid_argument("Unknown dtype name: " + name);
}

/// Optimization levels for model compilation
enum class OptLevel : uint8_t {
    None = 0,     // No optimization
    Basic = 1,    // Constant folding, dead code elimination
    Extended = 2, // + Operator fusion
    All = 3       // + Layout optimization, aggressive fusion
};

/// Tensor shape with optional dynamic dimensions
/// nullopt indicates a dynamic dimension
using Shape = std::vector<std::optional<int64_t>>;

/// Convert shape to string for debugging
inline std::string shape_to_string(const Shape& shape) {
    std::string result = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) result += ", ";
        if (shape[i].has_value()) {
            result += std::to_string(shape[i].value());
        } else {
            result += "?";
        }
    }
    result += "]";
    return result;
}

/// Check if shape has any dynamic dimensions
inline bool is_dynamic_shape(const Shape& shape) {
    for (const auto& dim : shape) {
        if (!dim.has_value()) return true;
    }
    return false;
}

/// Calculate total elements (returns nullopt if dynamic)
/// Security: Uses checked_product() to prevent integer overflow (CRIT-02 fix)
inline std::optional<int64_t> shape_num_elements(const Shape& shape) {
    if (is_dynamic_shape(shape)) return std::nullopt;
    if (shape.empty()) return 1;

    // Extract concrete dimensions for overflow-safe multiplication
    std::vector<int64_t> dims;
    dims.reserve(shape.size());
    for (const auto& dim : shape) {
        dims.push_back(dim.value());
    }

    // Use checked_product for overflow protection
    return checked_product(dims);
}

/// Convert concrete shape vector to Shape type
inline Shape to_shape(const std::vector<int64_t>& dims) {
    Shape result;
    result.reserve(dims.size());
    for (auto d : dims) {
        result.push_back(d);
    }
    return result;
}

/// Convert Shape to concrete dimensions (throws if dynamic)
inline std::vector<int64_t> to_dims(const Shape& shape) {
    std::vector<int64_t> result;
    result.reserve(shape.size());
    for (const auto& dim : shape) {
        if (!dim.has_value()) {
            throw std::runtime_error("Cannot convert dynamic shape to concrete dimensions");
        }
        result.push_back(dim.value());
    }
    return result;
}

/// Tensor metadata
struct TensorInfo {
    std::string name;
    Shape shape;
    DType dtype;

    TensorInfo() = default;
    TensorInfo(std::string n, Shape s, DType d)
        : name(std::move(n)), shape(std::move(s)), dtype(d) {}

    bool is_dynamic() const { return is_dynamic_shape(shape); }

    std::optional<int64_t> num_elements() const {
        return shape_num_elements(shape);
    }

    std::optional<size_t> size_bytes() const {
        auto elems = num_elements();
        if (!elems.has_value()) return std::nullopt;
        return static_cast<size_t>(elems.value()) * dtype_size(dtype);
    }
};

/// Input/output argument descriptor (ONNX Runtime API compatible)
struct NodeArg {
    std::string name;
    std::vector<std::optional<int64_t>> shape;
    std::string type_str;  // e.g., "tensor(float32)"

    static NodeArg from_tensor_info(const TensorInfo& info) {
        NodeArg arg;
        arg.name = info.name;
        arg.shape = info.shape;
        arg.type_str = "tensor(" + dtype_name(info.dtype) + ")";
        return arg;
    }
};

/// Model metadata container
struct ModelMetadata {
    std::string producer_name;
    std::string producer_version;
    std::string domain;
    std::string description;
    std::string graph_name;
    int64_t version = 0;
    std::unordered_map<std::string, std::string> custom_metadata;
};

} // namespace pyflame_rt
