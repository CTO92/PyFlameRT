#pragma once

// ONNX protobuf data structures
// These match the ONNX protobuf specification for parsing ONNX models
// https://github.com/onnx/onnx/blob/main/onnx/onnx.proto

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <variant>
#include <optional>

namespace pyflame_rt {
namespace import {
namespace onnx_proto {

// ============================================================================
// ONNX Data Types (TensorProto.DataType)
// ============================================================================

enum class TensorDataType : int32_t {
    UNDEFINED = 0,
    FLOAT = 1,
    UINT8 = 2,
    INT8 = 3,
    UINT16 = 4,
    INT16 = 5,
    INT32 = 6,
    INT64 = 7,
    STRING = 8,
    BOOL = 9,
    FLOAT16 = 10,
    DOUBLE = 11,
    UINT32 = 12,
    UINT64 = 13,
    COMPLEX64 = 14,
    COMPLEX128 = 15,
    BFLOAT16 = 16,
};

// ============================================================================
// Tensor Type Information
// ============================================================================

struct Dimension {
    std::optional<int64_t> dim_value;  // Concrete dimension
    std::string dim_param;              // Symbolic dimension name

    bool is_concrete() const { return dim_value.has_value(); }
    bool is_symbolic() const { return !dim_param.empty(); }
};

struct TensorShapeProto {
    std::vector<Dimension> dims;
};

struct TensorTypeProto {
    TensorDataType elem_type = TensorDataType::UNDEFINED;
    TensorShapeProto shape;
};

struct TypeProto {
    TensorTypeProto tensor_type;
    // Other type variants (sequence, map, etc.) not supported yet
};

// ============================================================================
// Value Information
// ============================================================================

struct ValueInfoProto {
    std::string name;
    TypeProto type;
    std::string doc_string;
};

// ============================================================================
// Tensor (weights/constants)
// ============================================================================

struct TensorProto {
    std::string name;
    std::vector<int64_t> dims;
    TensorDataType data_type = TensorDataType::UNDEFINED;

    // Raw data (preferred format)
    std::vector<uint8_t> raw_data;

    // Typed data (legacy format)
    std::vector<float> float_data;
    std::vector<int32_t> int32_data;
    std::vector<int64_t> int64_data;
    std::vector<double> double_data;
    std::vector<uint64_t> uint64_data;

    std::string doc_string;

    // External data location (for large tensors)
    std::string data_location;
    std::string external_data_path;
    int64_t external_data_offset = 0;
    int64_t external_data_length = 0;

    /// Get the number of elements in the tensor
    int64_t num_elements() const {
        int64_t total = 1;
        for (auto d : dims) {
            total *= d;
        }
        return total;
    }

    /// Get size in bytes
    size_t size_bytes() const {
        size_t elem_size = 1;
        switch (data_type) {
            case TensorDataType::FLOAT:
            case TensorDataType::INT32:
            case TensorDataType::UINT32:
                elem_size = 4;
                break;
            case TensorDataType::DOUBLE:
            case TensorDataType::INT64:
            case TensorDataType::UINT64:
            case TensorDataType::COMPLEX64:
                elem_size = 8;
                break;
            case TensorDataType::FLOAT16:
            case TensorDataType::BFLOAT16:
            case TensorDataType::INT16:
            case TensorDataType::UINT16:
                elem_size = 2;
                break;
            case TensorDataType::INT8:
            case TensorDataType::UINT8:
            case TensorDataType::BOOL:
                elem_size = 1;
                break;
            case TensorDataType::COMPLEX128:
                elem_size = 16;
                break;
            default:
                elem_size = 1;
                break;
        }
        return static_cast<size_t>(num_elements()) * elem_size;
    }
};

// ============================================================================
// Attribute
// ============================================================================

enum class AttributeType : int32_t {
    UNDEFINED = 0,
    FLOAT = 1,
    INT = 2,
    STRING = 3,
    TENSOR = 4,
    GRAPH = 5,
    FLOATS = 6,
    INTS = 7,
    STRINGS = 8,
    TENSORS = 9,
    GRAPHS = 10,
    SPARSE_TENSOR = 11,
    SPARSE_TENSORS = 12,
    TYPE_PROTO = 13,
    TYPE_PROTOS = 14,
};

// Forward declaration
struct GraphProto;

struct AttributeProto {
    std::string name;
    AttributeType type = AttributeType::UNDEFINED;
    std::string doc_string;

    // Single values
    float f = 0.0f;
    int64_t i = 0;
    std::string s;
    TensorProto t;

    // Repeated values
    std::vector<float> floats;
    std::vector<int64_t> ints;
    std::vector<std::string> strings;
    std::vector<TensorProto> tensors;

    // Subgraphs (for control flow)
    std::shared_ptr<GraphProto> g;
    std::vector<std::shared_ptr<GraphProto>> graphs;

    // Ref attribute name
    std::string ref_attr_name;
};

// ============================================================================
// Node (operator)
// ============================================================================

struct NodeProto {
    std::string name;
    std::string op_type;
    std::string domain;
    std::vector<std::string> input;
    std::vector<std::string> output;
    std::vector<AttributeProto> attribute;
    std::string doc_string;

    /// Get attribute by name
    const AttributeProto* get_attribute(const std::string& attr_name) const {
        for (const auto& attr : attribute) {
            if (attr.name == attr_name) {
                return &attr;
            }
        }
        return nullptr;
    }

    /// Check if attribute exists
    bool has_attribute(const std::string& attr_name) const {
        return get_attribute(attr_name) != nullptr;
    }
};

// ============================================================================
// Graph
// ============================================================================

struct GraphProto {
    std::string name;
    std::vector<NodeProto> node;
    std::vector<TensorProto> initializer;
    std::vector<TensorProto> sparse_initializer;
    std::vector<ValueInfoProto> input;
    std::vector<ValueInfoProto> output;
    std::vector<ValueInfoProto> value_info;  // Intermediate value shapes
    std::string doc_string;

    /// Get initializer by name
    const TensorProto* get_initializer(const std::string& name) const {
        for (const auto& init : initializer) {
            if (init.name == name) {
                return &init;
            }
        }
        return nullptr;
    }

    /// Check if name is an initializer
    bool is_initializer(const std::string& name) const {
        return get_initializer(name) != nullptr;
    }
};

// ============================================================================
// Opset Import
// ============================================================================

struct OperatorSetIdProto {
    std::string domain;  // "" for ONNX default domain
    int64_t version = 0;
};

// ============================================================================
// Model
// ============================================================================

struct ModelProto {
    int64_t ir_version = 0;
    std::vector<OperatorSetIdProto> opset_import;
    std::string producer_name;
    std::string producer_version;
    std::string domain;
    int64_t model_version = 0;
    std::string doc_string;
    GraphProto graph;
    std::vector<std::string> metadata_props_keys;
    std::vector<std::string> metadata_props_values;
    std::vector<std::string> training_info;  // Simplified
    std::vector<GraphProto> functions;  // Simplified

    /// Get opset version for a domain (empty string = default ONNX domain)
    int64_t get_opset_version(const std::string& domain = "") const {
        for (const auto& opset : opset_import) {
            if (opset.domain == domain) {
                return opset.version;
            }
        }
        return 0;
    }
};

} // namespace onnx_proto
} // namespace import
} // namespace pyflame_rt
