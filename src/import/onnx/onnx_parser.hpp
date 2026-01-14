#pragma once

#include "onnx_proto.hpp"
#include "pyflame_rt/errors.hpp"

#include <cstdint>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace pyflame_rt {
namespace import {

/// Parser for ONNX protobuf format
///
/// This is a lightweight protobuf parser that can read ONNX model files
/// without depending on the full protobuf library. It handles the binary
/// wire format directly.
///
class ONNXParser {
public:
    /// Parse ONNX model from file
    /// @param path Path to .onnx file
    /// @return Parsed model structure
    /// @throws ModelParseError on parse failure
    static onnx_proto::ModelProto parse_file(const std::string& path);

    /// Parse ONNX model from memory buffer
    /// @param data Pointer to model data
    /// @param size Size of data in bytes
    /// @return Parsed model structure
    /// @throws ModelParseError on parse failure
    static onnx_proto::ModelProto parse_buffer(const void* data, size_t size);

    /// Check if file is a valid ONNX model (quick check)
    /// @param path Path to file
    /// @return true if file appears to be ONNX format
    static bool is_onnx_file(const std::string& path);

    /// Get ONNX opset version from file without full parsing
    /// @param path Path to .onnx file
    /// @return Default opset version, or -1 on error
    static int get_opset_version(const std::string& path);

private:
    // Protobuf wire types
    enum WireType : uint8_t {
        VARINT = 0,
        FIXED64 = 1,
        LENGTH_DELIMITED = 2,
        START_GROUP = 3,  // Deprecated
        END_GROUP = 4,    // Deprecated
        FIXED32 = 5,
    };

    // Parser state
    struct ParseContext {
        const uint8_t* data;
        size_t size;
        size_t pos;
        int depth;  // Nesting depth (for debug/limits)

        static constexpr int MAX_DEPTH = 64;
        static constexpr size_t MAX_STRING_SIZE = 100 * 1024 * 1024;  // 100MB
        static constexpr size_t MAX_REPEATED_COUNT = 10000000;

        ParseContext(const void* buf, size_t len)
            : data(static_cast<const uint8_t*>(buf))
            , size(len)
            , pos(0)
            , depth(0) {}

        bool at_end() const { return pos >= size; }
        size_t remaining() const { return pos < size ? size - pos : 0; }

        void check_size(size_t needed, const std::string& context) const {
            if (remaining() < needed) {
                throw ModelParseError(
                    "Unexpected end of data while reading " + context,
                    "ONNX"
                );
            }
        }

        void check_depth() const {
            if (depth > MAX_DEPTH) {
                throw ModelParseError("Maximum nesting depth exceeded", "ONNX");
            }
        }
    };

    // Low-level protobuf reading
    static uint64_t read_varint(ParseContext& ctx);
    static uint32_t read_fixed32(ParseContext& ctx);
    static uint64_t read_fixed64(ParseContext& ctx);
    static std::string read_string(ParseContext& ctx, size_t len);
    static std::vector<uint8_t> read_bytes(ParseContext& ctx, size_t len);
    static void skip_field(ParseContext& ctx, WireType wire_type);

    // Field tag decoding
    static std::pair<uint32_t, WireType> read_tag(ParseContext& ctx);

    // High-level ONNX parsing
    static onnx_proto::ModelProto parse_model(ParseContext& ctx);
    static onnx_proto::GraphProto parse_graph(ParseContext& ctx);
    static onnx_proto::NodeProto parse_node(ParseContext& ctx);
    static onnx_proto::TensorProto parse_tensor(ParseContext& ctx);
    static onnx_proto::ValueInfoProto parse_value_info(ParseContext& ctx);
    static onnx_proto::TypeProto parse_type(ParseContext& ctx);
    static onnx_proto::TensorTypeProto parse_tensor_type(ParseContext& ctx);
    static onnx_proto::TensorShapeProto parse_tensor_shape(ParseContext& ctx);
    static onnx_proto::Dimension parse_dimension(ParseContext& ctx);
    static onnx_proto::AttributeProto parse_attribute(ParseContext& ctx);
    static onnx_proto::OperatorSetIdProto parse_opset_id(ParseContext& ctx);

    // Helpers
    static std::vector<int64_t> read_packed_int64(ParseContext& ctx, size_t len);
    static std::vector<float> read_packed_float(ParseContext& ctx, size_t len);
    static std::vector<double> read_packed_double(ParseContext& ctx, size_t len);
};

} // namespace import
} // namespace pyflame_rt
