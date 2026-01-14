#pragma once

// PyTorch Checkpoint Parser
// Parses Python pickle format with PyTorch tensor extensions
//
// PyTorch uses pickle protocol to serialize models. The checkpoint contains:
// - A dict mapping parameter names to tensors
// - Tensors are stored as _rebuild_tensor_v2 or similar rebuild functions
// - Storage objects contain the actual tensor data
//
// This parser handles the basic pickle opcodes needed for checkpoint loading.

#include "pyflame_rt/errors.hpp"
#include "pyflame_rt/tensor.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>
#include <memory>

namespace pyflame_rt {
namespace import {
namespace pytorch {

// ============================================================================
// Security Limits for Pickle Parsing
// ============================================================================

/// Security limits to prevent DoS attacks via malicious pickle files
struct PickleLimits {
    static constexpr size_t MAX_STRING_SIZE = 100 * 1024 * 1024;      // 100 MB max string
    static constexpr size_t MAX_BYTES_SIZE = 4ULL * 1024 * 1024 * 1024;  // 4 GB max bytes
    static constexpr size_t MAX_FILE_SIZE = 10ULL * 1024 * 1024 * 1024;  // 10 GB max file
    static constexpr size_t MAX_STACK_SIZE = 1000000;                  // 1M stack entries
    static constexpr size_t MAX_MEMO_SIZE = 1000000;                   // 1M memo entries
    static constexpr int MAX_RECURSION_DEPTH = 100;                    // Max nested structures
};

// ============================================================================
// Pickle Value Types
// ============================================================================

/// Represents a PyTorch tensor storage
struct TensorStorage {
    std::string dtype_name;  // e.g., "float", "long"
    size_t size = 0;         // Number of elements
    std::vector<uint8_t> data;

    // Storage location info
    std::string location;    // e.g., "cpu", "cuda:0"
    std::string key;         // Storage key in archive
};

/// Represents parsed tensor metadata
struct TensorMeta {
    std::vector<int64_t> shape;
    std::vector<int64_t> stride;
    int64_t storage_offset = 0;
    std::string dtype;
    bool requires_grad = false;

    std::shared_ptr<TensorStorage> storage;
};

// Forward declaration
struct PickleValue;

using PickleList = std::vector<std::shared_ptr<PickleValue>>;
using PickleDict = std::unordered_map<std::string, std::shared_ptr<PickleValue>>;
using PickleTuple = std::vector<std::shared_ptr<PickleValue>>;

/// Represents a value in the pickle stream
struct PickleValue {
    enum class Type {
        None,
        Bool,
        Int,
        Float,
        String,
        Bytes,
        List,
        Dict,
        Tuple,
        Tensor,
        Global,     // Python global reference (module.class)
        Reduce,     // Result of __reduce__ protocol
        PersId,     // Persistent ID reference
    };

    Type type = Type::None;

    // Value storage
    bool bool_val = false;
    int64_t int_val = 0;
    double float_val = 0.0;
    std::string string_val;
    std::vector<uint8_t> bytes_val;
    PickleList list_val;
    PickleDict dict_val;
    PickleTuple tuple_val;

    // Special types
    TensorMeta tensor_meta;

    // For Global type
    std::string module_name;
    std::string class_name;

    // Constructors for convenience
    static std::shared_ptr<PickleValue> make_none() {
        auto v = std::make_shared<PickleValue>();
        v->type = Type::None;
        return v;
    }

    static std::shared_ptr<PickleValue> make_bool(bool val) {
        auto v = std::make_shared<PickleValue>();
        v->type = Type::Bool;
        v->bool_val = val;
        return v;
    }

    static std::shared_ptr<PickleValue> make_int(int64_t val) {
        auto v = std::make_shared<PickleValue>();
        v->type = Type::Int;
        v->int_val = val;
        return v;
    }

    static std::shared_ptr<PickleValue> make_float(double val) {
        auto v = std::make_shared<PickleValue>();
        v->type = Type::Float;
        v->float_val = val;
        return v;
    }

    static std::shared_ptr<PickleValue> make_string(const std::string& val) {
        auto v = std::make_shared<PickleValue>();
        v->type = Type::String;
        v->string_val = val;
        return v;
    }

    static std::shared_ptr<PickleValue> make_bytes(const std::vector<uint8_t>& val) {
        auto v = std::make_shared<PickleValue>();
        v->type = Type::Bytes;
        v->bytes_val = val;
        return v;
    }

    static std::shared_ptr<PickleValue> make_list() {
        auto v = std::make_shared<PickleValue>();
        v->type = Type::List;
        return v;
    }

    static std::shared_ptr<PickleValue> make_dict() {
        auto v = std::make_shared<PickleValue>();
        v->type = Type::Dict;
        return v;
    }

    static std::shared_ptr<PickleValue> make_tuple() {
        auto v = std::make_shared<PickleValue>();
        v->type = Type::Tuple;
        return v;
    }

    /// Check if this is a tensor value
    bool is_tensor() const { return type == Type::Tensor; }

    /// Get as dict (throws if not a dict)
    const PickleDict& as_dict() const {
        if (type != Type::Dict) {
            throw std::runtime_error("Expected dict type");
        }
        return dict_val;
    }

    /// Get string value from dict key
    std::string get_string(const std::string& key) const {
        auto it = dict_val.find(key);
        if (it == dict_val.end() || it->second->type != Type::String) {
            return "";
        }
        return it->second->string_val;
    }
};

// ============================================================================
// PyTorch Checkpoint Parser
// ============================================================================

/// Parser for PyTorch checkpoint files
///
/// PyTorch checkpoints are ZIP archives containing:
/// - data.pkl: The pickled state dictionary
/// - data/N: Binary tensor storage files
///
/// This parser handles both the pickle format and the ZIP structure.
class PickleParser {
public:
    /// Result of parsing a checkpoint
    struct ParseResult {
        /// Parsed state dictionary (param_name -> tensor metadata)
        std::unordered_map<std::string, TensorMeta> state_dict;

        /// Model metadata if present
        std::unordered_map<std::string, std::string> metadata;

        /// Warnings during parsing
        std::vector<std::string> warnings;
    };

    /// Parse a PyTorch checkpoint file
    /// @param path Path to .pt/.pth file
    /// @return Parsed state dictionary
    /// @throws ModelParseError on parse failure
    static ParseResult parse_file(const std::string& path);

    /// Parse a PyTorch checkpoint from memory
    /// @param data Pointer to checkpoint data
    /// @param size Size of data in bytes
    /// @return Parsed state dictionary
    /// @throws ModelParseError on parse failure
    static ParseResult parse_buffer(const void* data, size_t size);

    /// Check if a file appears to be a PyTorch checkpoint
    static bool is_pytorch_checkpoint(const std::string& path);

private:
    // Pickle opcodes we need to handle
    enum Opcode : uint8_t {
        MARK = '(',
        STOP = '.',
        POP = '0',
        POP_MARK = '1',
        DUP = '2',
        FLOAT = 'F',
        INT = 'I',
        BININT = 'J',
        BININT1 = 'K',
        LONG = 'L',
        BININT2 = 'M',
        NONE = 'N',
        PERSID = 'P',
        BINPERSID = 'Q',
        REDUCE = 'R',
        STRING = 'S',
        BINSTRING = 'T',
        SHORT_BINSTRING = 'U',
        UNICODE = 'V',
        BINUNICODE = 'X',
        APPEND = 'a',
        BUILD = 'b',
        GLOBAL = 'c',
        DICT = 'd',
        EMPTY_DICT = '}',
        APPENDS = 'e',
        GET = 'g',
        BINGET = 'h',
        INST = 'i',
        LONG_BINGET = 'j',
        LIST = 'l',
        EMPTY_LIST = ']',
        OBJ = 'o',
        PUT = 'p',
        BINPUT = 'q',
        LONG_BINPUT = 'r',
        SETITEM = 's',
        TUPLE = 't',
        EMPTY_TUPLE = ')',
        SETITEMS = 'u',
        BINFLOAT = 'G',

        // Protocol 2+ opcodes
        PROTO = '\x80',
        NEWOBJ = '\x81',
        EXT1 = '\x82',
        EXT2 = '\x83',
        EXT4 = '\x84',
        TUPLE1 = '\x85',
        TUPLE2 = '\x86',
        TUPLE3 = '\x87',
        NEWTRUE = '\x88',
        NEWFALSE = '\x89',
        LONG1 = '\x8a',
        LONG4 = '\x8b',

        // Protocol 3+ opcodes
        BINBYTES = 'B',
        SHORT_BINBYTES = 'C',

        // Protocol 4+ opcodes
        SHORT_BINUNICODE = '\x8c',
        BINUNICODE8 = '\x8d',
        BINBYTES8 = '\x8e',
        EMPTY_SET = '\x8f',
        ADDITEMS = '\x90',
        FROZENSET = '\x91',
        NEWOBJ_EX = '\x92',
        STACK_GLOBAL = '\x93',
        MEMOIZE = '\x94',
        FRAME = '\x95',

        // Protocol 5+ opcodes
        BYTEARRAY8 = '\x96',
        NEXT_BUFFER = '\x97',
        READONLY_BUFFER = '\x98',
    };

    // Parse context
    struct ParseContext;

    // Internal parsing methods
    static std::shared_ptr<PickleValue> parse_pickle(ParseContext& ctx);
    static void execute_opcode(ParseContext& ctx, uint8_t opcode);

    // Helper methods for reading data
    static uint8_t read_byte(ParseContext& ctx);
    static int32_t read_int32(ParseContext& ctx);
    static int64_t read_int64(ParseContext& ctx);
    static uint32_t read_uint32(ParseContext& ctx);
    static double read_double(ParseContext& ctx);
    static std::string read_line(ParseContext& ctx);
    static std::string read_string(ParseContext& ctx, size_t len);
    static std::vector<uint8_t> read_bytes(ParseContext& ctx, size_t len);

    // ZIP archive handling
    static bool is_zip_file(const void* data, size_t size);
    static std::unordered_map<std::string, std::vector<uint8_t>> extract_zip(
        const void* data, size_t size);
};

// ============================================================================
// Tensor Conversion
// ============================================================================

/// Convert a parsed TensorMeta to PyFlameRT Tensor
/// @param meta Tensor metadata from pickle parser
/// @return PyFlameRT Tensor with the data
Tensor convert_pytorch_tensor(const TensorMeta& meta);

/// Get PyFlameRT DType from PyTorch dtype string
/// @param pytorch_dtype PyTorch dtype string (e.g., "Float", "Long")
/// @return Corresponding PyFlameRT DType
DType pytorch_dtype_to_dtype(const std::string& pytorch_dtype);

} // namespace pytorch
} // namespace import
} // namespace pyflame_rt
