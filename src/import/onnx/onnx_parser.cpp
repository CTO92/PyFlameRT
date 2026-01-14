#include "onnx_parser.hpp"

#include <cstring>
#include <fstream>
#include <limits>

namespace pyflame_rt {
namespace import {

// ============================================================================
// Public API
// ============================================================================

onnx_proto::ModelProto ONNXParser::parse_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw ModelParseError("Cannot open file: " + path, "ONNX");
    }

    auto size = file.tellg();
    if (size <= 0) {
        throw ModelParseError("Empty or invalid file: " + path, "ONNX");
    }

    // Safety limit: 2GB max file size
    constexpr size_t MAX_FILE_SIZE = 2ULL * 1024 * 1024 * 1024;
    if (static_cast<size_t>(size) > MAX_FILE_SIZE) {
        throw ModelParseError("File too large: " + path, "ONNX");
    }

    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> buffer(static_cast<size_t>(size));
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        throw ModelParseError("Failed to read file: " + path, "ONNX");
    }

    return parse_buffer(buffer.data(), buffer.size());
}

onnx_proto::ModelProto ONNXParser::parse_buffer(const void* data, size_t size) {
    if (!data || size == 0) {
        throw ModelParseError("Empty model buffer", "ONNX");
    }

    ParseContext ctx(data, size);
    return parse_model(ctx);
}

bool ONNXParser::is_onnx_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    // Check file extension
    size_t dot_pos = path.rfind('.');
    if (dot_pos != std::string::npos) {
        std::string ext = path.substr(dot_pos);
        if (ext == ".onnx") {
            return true;
        }
    }

    // Try to read first few bytes and check if it looks like protobuf
    uint8_t header[4];
    if (!file.read(reinterpret_cast<char*>(header), 4)) {
        return false;
    }

    // Protobuf files typically start with a field tag
    // Field 1 (ir_version) with wire type VARINT = (1 << 3) | 0 = 0x08
    // But this isn't a reliable check, so just trust the extension
    return header[0] == 0x08;
}

int ONNXParser::get_opset_version(const std::string& path) {
    try {
        auto model = parse_file(path);
        return static_cast<int>(model.get_opset_version());
    } catch (const std::exception&) {
        return -1;
    }
}

// ============================================================================
// Low-level protobuf reading
// ============================================================================

uint64_t ONNXParser::read_varint(ParseContext& ctx) {
    uint64_t result = 0;
    int shift = 0;

    while (true) {
        ctx.check_size(1, "varint");
        uint8_t byte = ctx.data[ctx.pos++];
        result |= static_cast<uint64_t>(byte & 0x7F) << shift;

        if ((byte & 0x80) == 0) {
            break;
        }

        shift += 7;
        if (shift > 63) {
            throw ModelParseError("Varint too large", "ONNX");
        }
    }

    return result;
}

uint32_t ONNXParser::read_fixed32(ParseContext& ctx) {
    ctx.check_size(4, "fixed32");
    uint32_t result;
    std::memcpy(&result, ctx.data + ctx.pos, 4);
    ctx.pos += 4;
    return result;
}

uint64_t ONNXParser::read_fixed64(ParseContext& ctx) {
    ctx.check_size(8, "fixed64");
    uint64_t result;
    std::memcpy(&result, ctx.data + ctx.pos, 8);
    ctx.pos += 8;
    return result;
}

std::string ONNXParser::read_string(ParseContext& ctx, size_t len) {
    if (len > ParseContext::MAX_STRING_SIZE) {
        throw ModelParseError("String too large: " + std::to_string(len), "ONNX");
    }
    ctx.check_size(len, "string");
    std::string result(reinterpret_cast<const char*>(ctx.data + ctx.pos), len);
    ctx.pos += len;
    return result;
}

std::vector<uint8_t> ONNXParser::read_bytes(ParseContext& ctx, size_t len) {
    if (len > ParseContext::MAX_STRING_SIZE) {
        throw ModelParseError("Bytes too large: " + std::to_string(len), "ONNX");
    }
    ctx.check_size(len, "bytes");
    std::vector<uint8_t> result(ctx.data + ctx.pos, ctx.data + ctx.pos + len);
    ctx.pos += len;
    return result;
}

void ONNXParser::skip_field(ParseContext& ctx, WireType wire_type) {
    switch (wire_type) {
        case VARINT:
            read_varint(ctx);
            break;
        case FIXED64:
            ctx.check_size(8, "skip fixed64");
            ctx.pos += 8;
            break;
        case LENGTH_DELIMITED: {
            auto len = read_varint(ctx);
            ctx.check_size(static_cast<size_t>(len), "skip length-delimited");
            ctx.pos += static_cast<size_t>(len);
            break;
        }
        case FIXED32:
            ctx.check_size(4, "skip fixed32");
            ctx.pos += 4;
            break;
        case START_GROUP:
        case END_GROUP:
            throw ModelParseError("Group wire types not supported", "ONNX");
        default:
            throw ModelParseError("Unknown wire type: " + std::to_string(wire_type), "ONNX");
    }
}

std::pair<uint32_t, ONNXParser::WireType> ONNXParser::read_tag(ParseContext& ctx) {
    auto tag = read_varint(ctx);
    return {
        static_cast<uint32_t>(tag >> 3),
        static_cast<WireType>(tag & 0x07)
    };
}

// ============================================================================
// Packed array reading
// ============================================================================

std::vector<int64_t> ONNXParser::read_packed_int64(ParseContext& ctx, size_t len) {
    std::vector<int64_t> result;
    result.reserve(len / 4);  // Rough estimate

    size_t end = ctx.pos + len;
    while (ctx.pos < end) {
        auto val = read_varint(ctx);
        result.push_back(static_cast<int64_t>(val));
    }
    return result;
}

std::vector<float> ONNXParser::read_packed_float(ParseContext& ctx, size_t len) {
    if (len % 4 != 0) {
        throw ModelParseError("Invalid packed float length", "ONNX");
    }
    size_t count = len / 4;
    std::vector<float> result(count);
    std::memcpy(result.data(), ctx.data + ctx.pos, len);
    ctx.pos += len;
    return result;
}

std::vector<double> ONNXParser::read_packed_double(ParseContext& ctx, size_t len) {
    if (len % 8 != 0) {
        throw ModelParseError("Invalid packed double length", "ONNX");
    }
    size_t count = len / 8;
    std::vector<double> result(count);
    std::memcpy(result.data(), ctx.data + ctx.pos, len);
    ctx.pos += len;
    return result;
}

// ============================================================================
// High-level ONNX parsing
// ============================================================================

// ONNX ModelProto field numbers (from onnx.proto3)
namespace ModelFields {
    constexpr uint32_t IR_VERSION = 1;
    constexpr uint32_t OPSET_IMPORT = 8;
    constexpr uint32_t PRODUCER_NAME = 2;
    constexpr uint32_t PRODUCER_VERSION = 3;
    constexpr uint32_t DOMAIN = 4;
    constexpr uint32_t MODEL_VERSION = 5;
    constexpr uint32_t DOC_STRING = 6;
    constexpr uint32_t GRAPH = 7;
    constexpr uint32_t METADATA_PROPS = 14;
    constexpr uint32_t TRAINING_INFO = 20;
    constexpr uint32_t FUNCTIONS = 25;
}

onnx_proto::ModelProto ONNXParser::parse_model(ParseContext& ctx) {
    onnx_proto::ModelProto model;

    while (!ctx.at_end()) {
        auto [field, wire_type] = read_tag(ctx);

        switch (field) {
            case ModelFields::IR_VERSION:
                model.ir_version = static_cast<int64_t>(read_varint(ctx));
                break;

            case ModelFields::OPSET_IMPORT: {
                auto len = read_varint(ctx);
                ParseContext sub_ctx(ctx.data + ctx.pos, static_cast<size_t>(len));
                sub_ctx.depth = ctx.depth + 1;
                sub_ctx.check_depth();
                model.opset_import.push_back(parse_opset_id(sub_ctx));
                ctx.pos += static_cast<size_t>(len);
                break;
            }

            case ModelFields::PRODUCER_NAME: {
                auto len = read_varint(ctx);
                model.producer_name = read_string(ctx, static_cast<size_t>(len));
                break;
            }

            case ModelFields::PRODUCER_VERSION: {
                auto len = read_varint(ctx);
                model.producer_version = read_string(ctx, static_cast<size_t>(len));
                break;
            }

            case ModelFields::DOMAIN: {
                auto len = read_varint(ctx);
                model.domain = read_string(ctx, static_cast<size_t>(len));
                break;
            }

            case ModelFields::MODEL_VERSION:
                model.model_version = static_cast<int64_t>(read_varint(ctx));
                break;

            case ModelFields::DOC_STRING: {
                auto len = read_varint(ctx);
                model.doc_string = read_string(ctx, static_cast<size_t>(len));
                break;
            }

            case ModelFields::GRAPH: {
                auto len = read_varint(ctx);
                ParseContext sub_ctx(ctx.data + ctx.pos, static_cast<size_t>(len));
                sub_ctx.depth = ctx.depth + 1;
                sub_ctx.check_depth();
                model.graph = parse_graph(sub_ctx);
                ctx.pos += static_cast<size_t>(len);
                break;
            }

            default:
                skip_field(ctx, wire_type);
                break;
        }
    }

    return model;
}

// ONNX GraphProto field numbers
namespace GraphFields {
    constexpr uint32_t NODE = 1;
    constexpr uint32_t NAME = 2;
    constexpr uint32_t INITIALIZER = 5;
    constexpr uint32_t SPARSE_INITIALIZER = 15;
    constexpr uint32_t DOC_STRING = 10;
    constexpr uint32_t INPUT = 11;
    constexpr uint32_t OUTPUT = 12;
    constexpr uint32_t VALUE_INFO = 13;
}

onnx_proto::GraphProto ONNXParser::parse_graph(ParseContext& ctx) {
    onnx_proto::GraphProto graph;

    while (!ctx.at_end()) {
        auto [field, wire_type] = read_tag(ctx);

        switch (field) {
            case GraphFields::NODE: {
                auto len = read_varint(ctx);
                ParseContext sub_ctx(ctx.data + ctx.pos, static_cast<size_t>(len));
                sub_ctx.depth = ctx.depth + 1;
                sub_ctx.check_depth();
                graph.node.push_back(parse_node(sub_ctx));
                ctx.pos += static_cast<size_t>(len);
                break;
            }

            case GraphFields::NAME: {
                auto len = read_varint(ctx);
                graph.name = read_string(ctx, static_cast<size_t>(len));
                break;
            }

            case GraphFields::INITIALIZER: {
                auto len = read_varint(ctx);
                ParseContext sub_ctx(ctx.data + ctx.pos, static_cast<size_t>(len));
                sub_ctx.depth = ctx.depth + 1;
                sub_ctx.check_depth();
                graph.initializer.push_back(parse_tensor(sub_ctx));
                ctx.pos += static_cast<size_t>(len);
                break;
            }

            case GraphFields::INPUT: {
                auto len = read_varint(ctx);
                ParseContext sub_ctx(ctx.data + ctx.pos, static_cast<size_t>(len));
                sub_ctx.depth = ctx.depth + 1;
                sub_ctx.check_depth();
                graph.input.push_back(parse_value_info(sub_ctx));
                ctx.pos += static_cast<size_t>(len);
                break;
            }

            case GraphFields::OUTPUT: {
                auto len = read_varint(ctx);
                ParseContext sub_ctx(ctx.data + ctx.pos, static_cast<size_t>(len));
                sub_ctx.depth = ctx.depth + 1;
                sub_ctx.check_depth();
                graph.output.push_back(parse_value_info(sub_ctx));
                ctx.pos += static_cast<size_t>(len);
                break;
            }

            case GraphFields::VALUE_INFO: {
                auto len = read_varint(ctx);
                ParseContext sub_ctx(ctx.data + ctx.pos, static_cast<size_t>(len));
                sub_ctx.depth = ctx.depth + 1;
                sub_ctx.check_depth();
                graph.value_info.push_back(parse_value_info(sub_ctx));
                ctx.pos += static_cast<size_t>(len);
                break;
            }

            case GraphFields::DOC_STRING: {
                auto len = read_varint(ctx);
                graph.doc_string = read_string(ctx, static_cast<size_t>(len));
                break;
            }

            default:
                skip_field(ctx, wire_type);
                break;
        }
    }

    return graph;
}

// ONNX NodeProto field numbers
namespace NodeFields {
    constexpr uint32_t INPUT = 1;
    constexpr uint32_t OUTPUT = 2;
    constexpr uint32_t NAME = 3;
    constexpr uint32_t OP_TYPE = 4;
    constexpr uint32_t DOMAIN = 7;
    constexpr uint32_t ATTRIBUTE = 5;
    constexpr uint32_t DOC_STRING = 6;
}

onnx_proto::NodeProto ONNXParser::parse_node(ParseContext& ctx) {
    onnx_proto::NodeProto node;

    while (!ctx.at_end()) {
        auto [field, wire_type] = read_tag(ctx);

        switch (field) {
            case NodeFields::INPUT: {
                auto len = read_varint(ctx);
                node.input.push_back(read_string(ctx, static_cast<size_t>(len)));
                break;
            }

            case NodeFields::OUTPUT: {
                auto len = read_varint(ctx);
                node.output.push_back(read_string(ctx, static_cast<size_t>(len)));
                break;
            }

            case NodeFields::NAME: {
                auto len = read_varint(ctx);
                node.name = read_string(ctx, static_cast<size_t>(len));
                break;
            }

            case NodeFields::OP_TYPE: {
                auto len = read_varint(ctx);
                node.op_type = read_string(ctx, static_cast<size_t>(len));
                break;
            }

            case NodeFields::DOMAIN: {
                auto len = read_varint(ctx);
                node.domain = read_string(ctx, static_cast<size_t>(len));
                break;
            }

            case NodeFields::ATTRIBUTE: {
                auto len = read_varint(ctx);
                ParseContext sub_ctx(ctx.data + ctx.pos, static_cast<size_t>(len));
                sub_ctx.depth = ctx.depth + 1;
                sub_ctx.check_depth();
                node.attribute.push_back(parse_attribute(sub_ctx));
                ctx.pos += static_cast<size_t>(len);
                break;
            }

            case NodeFields::DOC_STRING: {
                auto len = read_varint(ctx);
                node.doc_string = read_string(ctx, static_cast<size_t>(len));
                break;
            }

            default:
                skip_field(ctx, wire_type);
                break;
        }
    }

    return node;
}

// ONNX TensorProto field numbers
namespace TensorFields {
    constexpr uint32_t DIMS = 1;
    constexpr uint32_t DATA_TYPE = 2;
    constexpr uint32_t SEGMENT = 3;  // Deprecated
    constexpr uint32_t FLOAT_DATA = 4;
    constexpr uint32_t INT32_DATA = 5;
    constexpr uint32_t STRING_DATA = 6;
    constexpr uint32_t INT64_DATA = 7;
    constexpr uint32_t NAME = 8;
    constexpr uint32_t DOC_STRING = 12;
    constexpr uint32_t RAW_DATA = 9;
    constexpr uint32_t EXTERNAL_DATA = 13;
    constexpr uint32_t DATA_LOCATION = 14;
    constexpr uint32_t DOUBLE_DATA = 10;
    constexpr uint32_t UINT64_DATA = 11;
}

onnx_proto::TensorProto ONNXParser::parse_tensor(ParseContext& ctx) {
    onnx_proto::TensorProto tensor;

    while (!ctx.at_end()) {
        auto [field, wire_type] = read_tag(ctx);

        switch (field) {
            case TensorFields::DIMS: {
                if (wire_type == LENGTH_DELIMITED) {
                    // Packed format
                    auto len = read_varint(ctx);
                    tensor.dims = read_packed_int64(ctx, static_cast<size_t>(len));
                } else {
                    // Single value
                    tensor.dims.push_back(static_cast<int64_t>(read_varint(ctx)));
                }
                break;
            }

            case TensorFields::DATA_TYPE:
                tensor.data_type = static_cast<onnx_proto::TensorDataType>(read_varint(ctx));
                break;

            case TensorFields::FLOAT_DATA: {
                if (wire_type == LENGTH_DELIMITED) {
                    auto len = read_varint(ctx);
                    tensor.float_data = read_packed_float(ctx, static_cast<size_t>(len));
                } else {
                    tensor.float_data.push_back(*reinterpret_cast<const float*>(&read_fixed32(ctx)));
                }
                break;
            }

            case TensorFields::INT32_DATA: {
                if (wire_type == LENGTH_DELIMITED) {
                    auto len = read_varint(ctx);
                    auto vals = read_packed_int64(ctx, static_cast<size_t>(len));
                    for (auto v : vals) {
                        tensor.int32_data.push_back(static_cast<int32_t>(v));
                    }
                } else {
                    tensor.int32_data.push_back(static_cast<int32_t>(read_varint(ctx)));
                }
                break;
            }

            case TensorFields::INT64_DATA: {
                if (wire_type == LENGTH_DELIMITED) {
                    auto len = read_varint(ctx);
                    tensor.int64_data = read_packed_int64(ctx, static_cast<size_t>(len));
                } else {
                    tensor.int64_data.push_back(static_cast<int64_t>(read_varint(ctx)));
                }
                break;
            }

            case TensorFields::NAME: {
                auto len = read_varint(ctx);
                tensor.name = read_string(ctx, static_cast<size_t>(len));
                break;
            }

            case TensorFields::RAW_DATA: {
                auto len = read_varint(ctx);
                tensor.raw_data = read_bytes(ctx, static_cast<size_t>(len));
                break;
            }

            case TensorFields::DOUBLE_DATA: {
                if (wire_type == LENGTH_DELIMITED) {
                    auto len = read_varint(ctx);
                    tensor.double_data = read_packed_double(ctx, static_cast<size_t>(len));
                } else {
                    tensor.double_data.push_back(*reinterpret_cast<const double*>(&read_fixed64(ctx)));
                }
                break;
            }

            case TensorFields::DOC_STRING: {
                auto len = read_varint(ctx);
                tensor.doc_string = read_string(ctx, static_cast<size_t>(len));
                break;
            }

            default:
                skip_field(ctx, wire_type);
                break;
        }
    }

    return tensor;
}

// ONNX ValueInfoProto field numbers
namespace ValueInfoFields {
    constexpr uint32_t NAME = 1;
    constexpr uint32_t TYPE = 2;
    constexpr uint32_t DOC_STRING = 3;
}

onnx_proto::ValueInfoProto ONNXParser::parse_value_info(ParseContext& ctx) {
    onnx_proto::ValueInfoProto value_info;

    while (!ctx.at_end()) {
        auto [field, wire_type] = read_tag(ctx);

        switch (field) {
            case ValueInfoFields::NAME: {
                auto len = read_varint(ctx);
                value_info.name = read_string(ctx, static_cast<size_t>(len));
                break;
            }

            case ValueInfoFields::TYPE: {
                auto len = read_varint(ctx);
                ParseContext sub_ctx(ctx.data + ctx.pos, static_cast<size_t>(len));
                sub_ctx.depth = ctx.depth + 1;
                sub_ctx.check_depth();
                value_info.type = parse_type(sub_ctx);
                ctx.pos += static_cast<size_t>(len);
                break;
            }

            case ValueInfoFields::DOC_STRING: {
                auto len = read_varint(ctx);
                value_info.doc_string = read_string(ctx, static_cast<size_t>(len));
                break;
            }

            default:
                skip_field(ctx, wire_type);
                break;
        }
    }

    return value_info;
}

// ONNX TypeProto field numbers
namespace TypeFields {
    constexpr uint32_t TENSOR_TYPE = 1;
    // Other types (sequence, map, etc.) not supported yet
}

onnx_proto::TypeProto ONNXParser::parse_type(ParseContext& ctx) {
    onnx_proto::TypeProto type_proto;

    while (!ctx.at_end()) {
        auto [field, wire_type] = read_tag(ctx);

        switch (field) {
            case TypeFields::TENSOR_TYPE: {
                auto len = read_varint(ctx);
                ParseContext sub_ctx(ctx.data + ctx.pos, static_cast<size_t>(len));
                sub_ctx.depth = ctx.depth + 1;
                sub_ctx.check_depth();
                type_proto.tensor_type = parse_tensor_type(sub_ctx);
                ctx.pos += static_cast<size_t>(len);
                break;
            }

            default:
                skip_field(ctx, wire_type);
                break;
        }
    }

    return type_proto;
}

// ONNX TensorTypeProto field numbers
namespace TensorTypeFields {
    constexpr uint32_t ELEM_TYPE = 1;
    constexpr uint32_t SHAPE = 2;
}

onnx_proto::TensorTypeProto ONNXParser::parse_tensor_type(ParseContext& ctx) {
    onnx_proto::TensorTypeProto tensor_type;

    while (!ctx.at_end()) {
        auto [field, wire_type] = read_tag(ctx);

        switch (field) {
            case TensorTypeFields::ELEM_TYPE:
                tensor_type.elem_type = static_cast<onnx_proto::TensorDataType>(read_varint(ctx));
                break;

            case TensorTypeFields::SHAPE: {
                auto len = read_varint(ctx);
                ParseContext sub_ctx(ctx.data + ctx.pos, static_cast<size_t>(len));
                sub_ctx.depth = ctx.depth + 1;
                sub_ctx.check_depth();
                tensor_type.shape = parse_tensor_shape(sub_ctx);
                ctx.pos += static_cast<size_t>(len);
                break;
            }

            default:
                skip_field(ctx, wire_type);
                break;
        }
    }

    return tensor_type;
}

// ONNX TensorShapeProto field numbers
namespace TensorShapeFields {
    constexpr uint32_t DIM = 1;
}

onnx_proto::TensorShapeProto ONNXParser::parse_tensor_shape(ParseContext& ctx) {
    onnx_proto::TensorShapeProto shape;

    while (!ctx.at_end()) {
        auto [field, wire_type] = read_tag(ctx);

        switch (field) {
            case TensorShapeFields::DIM: {
                auto len = read_varint(ctx);
                ParseContext sub_ctx(ctx.data + ctx.pos, static_cast<size_t>(len));
                sub_ctx.depth = ctx.depth + 1;
                sub_ctx.check_depth();
                shape.dims.push_back(parse_dimension(sub_ctx));
                ctx.pos += static_cast<size_t>(len);
                break;
            }

            default:
                skip_field(ctx, wire_type);
                break;
        }
    }

    return shape;
}

// ONNX Dimension field numbers
namespace DimensionFields {
    constexpr uint32_t DIM_VALUE = 1;
    constexpr uint32_t DIM_PARAM = 2;
}

onnx_proto::Dimension ONNXParser::parse_dimension(ParseContext& ctx) {
    onnx_proto::Dimension dim;

    while (!ctx.at_end()) {
        auto [field, wire_type] = read_tag(ctx);

        switch (field) {
            case DimensionFields::DIM_VALUE:
                dim.dim_value = static_cast<int64_t>(read_varint(ctx));
                break;

            case DimensionFields::DIM_PARAM: {
                auto len = read_varint(ctx);
                dim.dim_param = read_string(ctx, static_cast<size_t>(len));
                break;
            }

            default:
                skip_field(ctx, wire_type);
                break;
        }
    }

    return dim;
}

// ONNX AttributeProto field numbers
namespace AttributeFields {
    constexpr uint32_t NAME = 1;
    constexpr uint32_t REF_ATTR_NAME = 21;
    constexpr uint32_t DOC_STRING = 13;
    constexpr uint32_t TYPE = 20;
    constexpr uint32_t F = 2;
    constexpr uint32_t I = 3;
    constexpr uint32_t S = 4;
    constexpr uint32_t T = 5;
    constexpr uint32_t G = 6;
    constexpr uint32_t FLOATS = 7;
    constexpr uint32_t INTS = 8;
    constexpr uint32_t STRINGS = 9;
    constexpr uint32_t TENSORS = 10;
    constexpr uint32_t GRAPHS = 11;
}

onnx_proto::AttributeProto ONNXParser::parse_attribute(ParseContext& ctx) {
    onnx_proto::AttributeProto attr;

    while (!ctx.at_end()) {
        auto [field, wire_type] = read_tag(ctx);

        switch (field) {
            case AttributeFields::NAME: {
                auto len = read_varint(ctx);
                attr.name = read_string(ctx, static_cast<size_t>(len));
                break;
            }

            case AttributeFields::TYPE:
                attr.type = static_cast<onnx_proto::AttributeType>(read_varint(ctx));
                break;

            case AttributeFields::F:
                attr.f = *reinterpret_cast<const float*>(&read_fixed32(ctx));
                break;

            case AttributeFields::I:
                attr.i = static_cast<int64_t>(read_varint(ctx));
                break;

            case AttributeFields::S: {
                auto len = read_varint(ctx);
                attr.s = read_string(ctx, static_cast<size_t>(len));
                break;
            }

            case AttributeFields::T: {
                auto len = read_varint(ctx);
                ParseContext sub_ctx(ctx.data + ctx.pos, static_cast<size_t>(len));
                sub_ctx.depth = ctx.depth + 1;
                sub_ctx.check_depth();
                attr.t = parse_tensor(sub_ctx);
                ctx.pos += static_cast<size_t>(len);
                break;
            }

            case AttributeFields::G: {
                auto len = read_varint(ctx);
                ParseContext sub_ctx(ctx.data + ctx.pos, static_cast<size_t>(len));
                sub_ctx.depth = ctx.depth + 1;
                sub_ctx.check_depth();
                attr.g = std::make_shared<onnx_proto::GraphProto>(parse_graph(sub_ctx));
                ctx.pos += static_cast<size_t>(len);
                break;
            }

            case AttributeFields::FLOATS: {
                if (wire_type == LENGTH_DELIMITED) {
                    auto len = read_varint(ctx);
                    attr.floats = read_packed_float(ctx, static_cast<size_t>(len));
                } else {
                    attr.floats.push_back(*reinterpret_cast<const float*>(&read_fixed32(ctx)));
                }
                break;
            }

            case AttributeFields::INTS: {
                if (wire_type == LENGTH_DELIMITED) {
                    auto len = read_varint(ctx);
                    attr.ints = read_packed_int64(ctx, static_cast<size_t>(len));
                } else {
                    attr.ints.push_back(static_cast<int64_t>(read_varint(ctx)));
                }
                break;
            }

            case AttributeFields::STRINGS: {
                auto len = read_varint(ctx);
                attr.strings.push_back(read_string(ctx, static_cast<size_t>(len)));
                break;
            }

            case AttributeFields::DOC_STRING: {
                auto len = read_varint(ctx);
                attr.doc_string = read_string(ctx, static_cast<size_t>(len));
                break;
            }

            case AttributeFields::REF_ATTR_NAME: {
                auto len = read_varint(ctx);
                attr.ref_attr_name = read_string(ctx, static_cast<size_t>(len));
                break;
            }

            default:
                skip_field(ctx, wire_type);
                break;
        }
    }

    return attr;
}

// ONNX OperatorSetIdProto field numbers
namespace OpsetIdFields {
    constexpr uint32_t DOMAIN = 1;
    constexpr uint32_t VERSION = 2;
}

onnx_proto::OperatorSetIdProto ONNXParser::parse_opset_id(ParseContext& ctx) {
    onnx_proto::OperatorSetIdProto opset;

    while (!ctx.at_end()) {
        auto [field, wire_type] = read_tag(ctx);

        switch (field) {
            case OpsetIdFields::DOMAIN: {
                auto len = read_varint(ctx);
                opset.domain = read_string(ctx, static_cast<size_t>(len));
                break;
            }

            case OpsetIdFields::VERSION:
                opset.version = static_cast<int64_t>(read_varint(ctx));
                break;

            default:
                skip_field(ctx, wire_type);
                break;
        }
    }

    return opset;
}

} // namespace import
} // namespace pyflame_rt
