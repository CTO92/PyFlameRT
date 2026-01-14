#include "pickle_parser.hpp"
#include "pyflame_rt/types.hpp"

#include <cstring>
#include <fstream>
#include <sstream>
#include <stack>
#include <algorithm>

namespace pyflame_rt {
namespace import {
namespace pytorch {

// ============================================================================
// Parse Context
// ============================================================================

struct PickleParser::ParseContext {
    const uint8_t* data;
    size_t size;
    size_t pos;

    // Pickle stack and memo
    std::vector<std::shared_ptr<PickleValue>> stack;
    std::vector<size_t> marks;  // Mark positions
    std::unordered_map<size_t, std::shared_ptr<PickleValue>> memo;

    // External data (tensor storage)
    std::unordered_map<std::string, std::vector<uint8_t>> storage_data;

    // Warnings
    std::vector<std::string> warnings;

    ParseContext(const void* buf, size_t len)
        : data(static_cast<const uint8_t*>(buf))
        , size(len)
        , pos(0) {}

    bool at_end() const { return pos >= size; }
    size_t remaining() const { return pos < size ? size - pos : 0; }

    void push(std::shared_ptr<PickleValue> val) {
        // Security: validate stack size limit
        if (stack.size() >= PickleLimits::MAX_STACK_SIZE) {
            throw ModelParseError("Pickle stack size limit exceeded", "PyTorch");
        }
        stack.push_back(std::move(val));
    }

    std::shared_ptr<PickleValue> pop() {
        if (stack.empty()) {
            throw ModelParseError("Stack underflow", "PyTorch");
        }
        auto val = std::move(stack.back());
        stack.pop_back();
        return val;
    }

    std::shared_ptr<PickleValue> top() {
        if (stack.empty()) {
            throw ModelParseError("Stack empty", "PyTorch");
        }
        return stack.back();
    }

    void push_mark() {
        marks.push_back(stack.size());
    }

    std::vector<std::shared_ptr<PickleValue>> pop_mark() {
        if (marks.empty()) {
            throw ModelParseError("No mark to pop", "PyTorch");
        }
        size_t mark = marks.back();
        marks.pop_back();

        std::vector<std::shared_ptr<PickleValue>> items;
        while (stack.size() > mark) {
            items.push_back(pop());
        }
        std::reverse(items.begin(), items.end());
        return items;
    }
};

// ============================================================================
// Helper Functions
// ============================================================================

uint8_t PickleParser::read_byte(ParseContext& ctx) {
    if (ctx.pos >= ctx.size) {
        throw ModelParseError("Unexpected end of data", "PyTorch");
    }
    return ctx.data[ctx.pos++];
}

int32_t PickleParser::read_int32(ParseContext& ctx) {
    if (ctx.remaining() < 4) {
        throw ModelParseError("Unexpected end of data reading int32", "PyTorch");
    }
    int32_t val;
    std::memcpy(&val, ctx.data + ctx.pos, 4);
    ctx.pos += 4;
    return val;
}

int64_t PickleParser::read_int64(ParseContext& ctx) {
    if (ctx.remaining() < 8) {
        throw ModelParseError("Unexpected end of data reading int64", "PyTorch");
    }
    int64_t val;
    std::memcpy(&val, ctx.data + ctx.pos, 8);
    ctx.pos += 8;
    return val;
}

uint32_t PickleParser::read_uint32(ParseContext& ctx) {
    if (ctx.remaining() < 4) {
        throw ModelParseError("Unexpected end of data reading uint32", "PyTorch");
    }
    uint32_t val;
    std::memcpy(&val, ctx.data + ctx.pos, 4);
    ctx.pos += 4;
    return val;
}

double PickleParser::read_double(ParseContext& ctx) {
    if (ctx.remaining() < 8) {
        throw ModelParseError("Unexpected end of data reading double", "PyTorch");
    }
    // IEEE 754 double is stored big-endian in pickle
    uint8_t bytes[8];
    for (int i = 0; i < 8; ++i) {
        bytes[7 - i] = ctx.data[ctx.pos++];
    }
    double val;
    std::memcpy(&val, bytes, 8);
    return val;
}

std::string PickleParser::read_line(ParseContext& ctx) {
    std::string line;
    while (ctx.pos < ctx.size) {
        char c = static_cast<char>(ctx.data[ctx.pos++]);
        if (c == '\n') {
            break;
        }
        line += c;
    }
    // Remove trailing \r if present
    if (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }
    return line;
}

std::string PickleParser::read_string(ParseContext& ctx, size_t len) {
    // Security: validate string size limit (CRIT-02 fix)
    if (len > PickleLimits::MAX_STRING_SIZE) {
        throw ModelParseError(
            "String size exceeds limit: " + std::to_string(len) +
            " > " + std::to_string(PickleLimits::MAX_STRING_SIZE), "PyTorch");
    }
    if (ctx.remaining() < len) {
        throw ModelParseError("Unexpected end of data reading string", "PyTorch");
    }
    std::string val(reinterpret_cast<const char*>(ctx.data + ctx.pos), len);
    ctx.pos += len;
    return val;
}

std::vector<uint8_t> PickleParser::read_bytes(ParseContext& ctx, size_t len) {
    // Security: validate bytes size limit (CRIT-02 fix)
    if (len > PickleLimits::MAX_BYTES_SIZE) {
        throw ModelParseError(
            "Bytes size exceeds limit: " + std::to_string(len) +
            " > " + std::to_string(PickleLimits::MAX_BYTES_SIZE), "PyTorch");
    }
    if (ctx.remaining() < len) {
        throw ModelParseError("Unexpected end of data reading bytes", "PyTorch");
    }
    std::vector<uint8_t> val(ctx.data + ctx.pos, ctx.data + ctx.pos + len);
    ctx.pos += len;
    return val;
}

// ============================================================================
// ZIP File Handling
// ============================================================================

bool PickleParser::is_zip_file(const void* data, size_t size) {
    // ZIP files start with PK\x03\x04
    if (size < 4) return false;
    const uint8_t* bytes = static_cast<const uint8_t*>(data);
    return bytes[0] == 'P' && bytes[1] == 'K' && bytes[2] == 0x03 && bytes[3] == 0x04;
}

// Simple ZIP extraction (handles uncompressed files only)
// PyTorch typically uses uncompressed ZIP for checkpoints
std::unordered_map<std::string, std::vector<uint8_t>> PickleParser::extract_zip(
    const void* data, size_t size) {

    std::unordered_map<std::string, std::vector<uint8_t>> files;
    const uint8_t* bytes = static_cast<const uint8_t*>(data);
    size_t pos = 0;

    // Parse local file headers
    while (pos + 30 <= size) {
        // Check for local file header signature
        if (bytes[pos] != 'P' || bytes[pos + 1] != 'K' ||
            bytes[pos + 2] != 0x03 || bytes[pos + 3] != 0x04) {
            break;  // End of local headers
        }

        // Read header fields
        uint16_t compression;
        std::memcpy(&compression, bytes + pos + 8, 2);

        uint32_t compressed_size;
        std::memcpy(&compressed_size, bytes + pos + 18, 4);

        uint32_t uncompressed_size;
        std::memcpy(&uncompressed_size, bytes + pos + 22, 4);

        uint16_t name_len;
        std::memcpy(&name_len, bytes + pos + 26, 2);

        uint16_t extra_len;
        std::memcpy(&extra_len, bytes + pos + 28, 2);

        // HIGH-02 fix: Use checked_add for all offset calculations to prevent overflow
        // Step 1: Calculate header_end = pos + 30 + name_len
        size_t header_end;
        try {
            header_end = checked_add(checked_add(pos, 30), static_cast<size_t>(name_len));
        } catch (const std::overflow_error&) {
            throw ModelParseError("Invalid ZIP: header offset overflow", "PyTorch");
        }

        // Step 2: Validate header_end is within bounds
        if (header_end > size) {
            throw ModelParseError("Invalid ZIP: filename extends past end", "PyTorch");
        }
        std::string filename(reinterpret_cast<const char*>(bytes + pos + 30), name_len);

        // Step 3: Calculate data_start = header_end + extra_len
        size_t data_start;
        try {
            data_start = checked_add(header_end, static_cast<size_t>(extra_len));
        } catch (const std::overflow_error&) {
            throw ModelParseError("Invalid ZIP: data offset overflow", "PyTorch");
        }

        size_t data_len = (compression == 0) ? uncompressed_size : compressed_size;

        // Step 4: Calculate data_end = data_start + data_len
        size_t data_end;
        try {
            data_end = checked_add(data_start, data_len);
        } catch (const std::overflow_error&) {
            throw ModelParseError("Invalid ZIP: data size overflow", "PyTorch");
        }

        // Step 5: Validate data range is within bounds
        if (data_end > size) {
            throw ModelParseError("Invalid ZIP: file data extends past end", "PyTorch");
        }

        // Security: validate data_len against limits
        if (data_len > PickleLimits::MAX_BYTES_SIZE) {
            throw ModelParseError("Invalid ZIP: file too large", "PyTorch");
        }

        // Extract file data (only uncompressed supported)
        if (compression == 0) {
            files[filename] = std::vector<uint8_t>(
                bytes + data_start,
                bytes + data_start + data_len
            );
        } else {
            // Skip compressed files with warning
            // In practice, PyTorch checkpoints are uncompressed
        }

        pos = data_start + data_len;
    }

    return files;
}

// ============================================================================
// Opcode Execution
// ============================================================================

void PickleParser::execute_opcode(ParseContext& ctx, uint8_t opcode) {
    switch (opcode) {
        case PROTO: {
            // Protocol version - just read and skip
            read_byte(ctx);
            break;
        }

        case FRAME: {
            // Frame size (protocol 4+)
            read_int64(ctx);
            break;
        }

        case MARK: {
            ctx.push_mark();
            break;
        }

        case STOP: {
            // End of pickle - handled in parse_pickle
            break;
        }

        case NONE: {
            ctx.push(PickleValue::make_none());
            break;
        }

        case NEWTRUE: {
            ctx.push(PickleValue::make_bool(true));
            break;
        }

        case NEWFALSE: {
            ctx.push(PickleValue::make_bool(false));
            break;
        }

        case INT: {
            std::string line = read_line(ctx);
            // Security: wrap stoll in try-catch (HIGH-01 fix)
            int64_t val;
            try {
                val = std::stoll(line);
            } catch (const std::exception& e) {
                throw ModelParseError("Invalid integer in pickle: '" + line + "'", "PyTorch");
            }
            ctx.push(PickleValue::make_int(val));
            break;
        }

        case BININT: {
            int32_t val = read_int32(ctx);
            ctx.push(PickleValue::make_int(val));
            break;
        }

        case BININT1: {
            uint8_t val = read_byte(ctx);
            ctx.push(PickleValue::make_int(val));
            break;
        }

        case BININT2: {
            uint8_t b0 = read_byte(ctx);
            uint8_t b1 = read_byte(ctx);
            uint16_t val = static_cast<uint16_t>(b0) | (static_cast<uint16_t>(b1) << 8);
            ctx.push(PickleValue::make_int(val));
            break;
        }

        case LONG: {
            std::string line = read_line(ctx);
            // Remove trailing 'L' if present
            if (!line.empty() && line.back() == 'L') {
                line.pop_back();
            }
            // Security: wrap stoll in try-catch (HIGH-01 fix)
            int64_t val;
            try {
                val = std::stoll(line);
            } catch (const std::exception& e) {
                throw ModelParseError("Invalid long integer in pickle: '" + line + "'", "PyTorch");
            }
            ctx.push(PickleValue::make_int(val));
            break;
        }

        case LONG1: {
            uint8_t n = read_byte(ctx);
            auto bytes = read_bytes(ctx, n);
            // Interpret as little-endian signed integer
            int64_t val = 0;
            for (size_t i = 0; i < bytes.size(); ++i) {
                val |= static_cast<int64_t>(bytes[i]) << (i * 8);
            }
            // Sign extend if needed
            if (n > 0 && (bytes[n - 1] & 0x80)) {
                for (size_t i = n; i < 8; ++i) {
                    val |= static_cast<int64_t>(0xFF) << (i * 8);
                }
            }
            ctx.push(PickleValue::make_int(val));
            break;
        }

        case FLOAT: {
            std::string line = read_line(ctx);
            // Security: wrap stod in try-catch (HIGH-01 fix)
            double val;
            try {
                val = std::stod(line);
            } catch (const std::exception& e) {
                throw ModelParseError("Invalid float in pickle: '" + line + "'", "PyTorch");
            }
            ctx.push(PickleValue::make_float(val));
            break;
        }

        case BINFLOAT: {
            double val = read_double(ctx);
            ctx.push(PickleValue::make_float(val));
            break;
        }

        case STRING: {
            std::string line = read_line(ctx);
            // Remove quotes
            if (line.size() >= 2 && line.front() == '\'' && line.back() == '\'') {
                line = line.substr(1, line.size() - 2);
            }
            ctx.push(PickleValue::make_string(line));
            break;
        }

        case BINSTRING: {
            int32_t len = read_int32(ctx);
            // Security: validate non-negative length (CRIT-01 fix)
            if (len < 0) {
                throw ModelParseError("Invalid negative string length in pickle: " +
                    std::to_string(len), "PyTorch");
            }
            std::string val = read_string(ctx, static_cast<size_t>(len));
            ctx.push(PickleValue::make_string(val));
            break;
        }

        case SHORT_BINSTRING: {
            uint8_t len = read_byte(ctx);
            std::string val = read_string(ctx, len);
            ctx.push(PickleValue::make_string(val));
            break;
        }

        case UNICODE: {
            std::string line = read_line(ctx);
            ctx.push(PickleValue::make_string(line));
            break;
        }

        case BINUNICODE: {
            uint32_t len = read_uint32(ctx);
            std::string val = read_string(ctx, len);
            ctx.push(PickleValue::make_string(val));
            break;
        }

        case SHORT_BINUNICODE: {
            uint8_t len = read_byte(ctx);
            std::string val = read_string(ctx, len);
            ctx.push(PickleValue::make_string(val));
            break;
        }

        case BINBYTES: {
            uint32_t len = read_uint32(ctx);
            auto val = read_bytes(ctx, len);
            ctx.push(PickleValue::make_bytes(val));
            break;
        }

        case SHORT_BINBYTES: {
            uint8_t len = read_byte(ctx);
            auto val = read_bytes(ctx, len);
            ctx.push(PickleValue::make_bytes(val));
            break;
        }

        case EMPTY_LIST: {
            ctx.push(PickleValue::make_list());
            break;
        }

        case LIST: {
            auto items = ctx.pop_mark();
            auto list = PickleValue::make_list();
            list->list_val = std::move(items);
            ctx.push(std::move(list));
            break;
        }

        case APPEND: {
            auto val = ctx.pop();
            auto list = ctx.top();
            if (list->type == PickleValue::Type::List) {
                list->list_val.push_back(std::move(val));
            }
            break;
        }

        case APPENDS: {
            auto items = ctx.pop_mark();
            auto list = ctx.top();
            if (list->type == PickleValue::Type::List) {
                for (auto& item : items) {
                    list->list_val.push_back(std::move(item));
                }
            }
            break;
        }

        case EMPTY_DICT: {
            ctx.push(PickleValue::make_dict());
            break;
        }

        case DICT: {
            auto items = ctx.pop_mark();
            auto dict = PickleValue::make_dict();
            for (size_t i = 0; i + 1 < items.size(); i += 2) {
                if (items[i]->type == PickleValue::Type::String) {
                    dict->dict_val[items[i]->string_val] = items[i + 1];
                }
            }
            ctx.push(std::move(dict));
            break;
        }

        case SETITEM: {
            auto val = ctx.pop();
            auto key = ctx.pop();
            auto dict = ctx.top();
            if (dict->type == PickleValue::Type::Dict &&
                key->type == PickleValue::Type::String) {
                dict->dict_val[key->string_val] = std::move(val);
            }
            break;
        }

        case SETITEMS: {
            auto items = ctx.pop_mark();
            auto dict = ctx.top();
            if (dict->type == PickleValue::Type::Dict) {
                for (size_t i = 0; i + 1 < items.size(); i += 2) {
                    if (items[i]->type == PickleValue::Type::String) {
                        dict->dict_val[items[i]->string_val] = items[i + 1];
                    }
                }
            }
            break;
        }

        case EMPTY_TUPLE: {
            ctx.push(PickleValue::make_tuple());
            break;
        }

        case TUPLE: {
            auto items = ctx.pop_mark();
            auto tuple = PickleValue::make_tuple();
            tuple->tuple_val = std::move(items);
            ctx.push(std::move(tuple));
            break;
        }

        case TUPLE1: {
            auto item = ctx.pop();
            auto tuple = PickleValue::make_tuple();
            tuple->tuple_val.push_back(std::move(item));
            ctx.push(std::move(tuple));
            break;
        }

        case TUPLE2: {
            auto item2 = ctx.pop();
            auto item1 = ctx.pop();
            auto tuple = PickleValue::make_tuple();
            tuple->tuple_val.push_back(std::move(item1));
            tuple->tuple_val.push_back(std::move(item2));
            ctx.push(std::move(tuple));
            break;
        }

        case TUPLE3: {
            auto item3 = ctx.pop();
            auto item2 = ctx.pop();
            auto item1 = ctx.pop();
            auto tuple = PickleValue::make_tuple();
            tuple->tuple_val.push_back(std::move(item1));
            tuple->tuple_val.push_back(std::move(item2));
            tuple->tuple_val.push_back(std::move(item3));
            ctx.push(std::move(tuple));
            break;
        }

        case GLOBAL: {
            std::string module = read_line(ctx);
            std::string name = read_line(ctx);
            auto val = std::make_shared<PickleValue>();
            val->type = PickleValue::Type::Global;
            val->module_name = module;
            val->class_name = name;
            ctx.push(std::move(val));
            break;
        }

        case STACK_GLOBAL: {
            auto name = ctx.pop();
            auto module = ctx.pop();
            auto val = std::make_shared<PickleValue>();
            val->type = PickleValue::Type::Global;
            if (module->type == PickleValue::Type::String) {
                val->module_name = module->string_val;
            }
            if (name->type == PickleValue::Type::String) {
                val->class_name = name->string_val;
            }
            ctx.push(std::move(val));
            break;
        }

        case REDUCE: {
            auto args = ctx.pop();
            auto callable = ctx.pop();

            // Handle PyTorch tensor reconstruction
            if (callable->type == PickleValue::Type::Global) {
                if (callable->class_name == "_rebuild_tensor_v2" ||
                    callable->class_name == "_rebuild_tensor") {
                    // This is a tensor
                    auto tensor_val = std::make_shared<PickleValue>();
                    tensor_val->type = PickleValue::Type::Tensor;

                    // Parse tensor args
                    if (args->type == PickleValue::Type::Tuple &&
                        args->tuple_val.size() >= 4) {

                        // args: (storage, offset, shape, stride, ...)
                        auto& storage = args->tuple_val[0];
                        auto& offset = args->tuple_val[1];
                        auto& shape = args->tuple_val[2];
                        auto& stride = args->tuple_val[3];

                        if (offset->type == PickleValue::Type::Int) {
                            tensor_val->tensor_meta.storage_offset = offset->int_val;
                        }

                        if (shape->type == PickleValue::Type::Tuple) {
                            for (auto& dim : shape->tuple_val) {
                                if (dim->type == PickleValue::Type::Int) {
                                    tensor_val->tensor_meta.shape.push_back(dim->int_val);
                                }
                            }
                        }

                        if (stride->type == PickleValue::Type::Tuple) {
                            for (auto& s : stride->tuple_val) {
                                if (s->type == PickleValue::Type::Int) {
                                    tensor_val->tensor_meta.stride.push_back(s->int_val);
                                }
                            }
                        }

                        // Storage contains tensor data
                        if (storage->type == PickleValue::Type::Reduce ||
                            storage->type == PickleValue::Type::PersId) {
                            tensor_val->tensor_meta.storage = storage->tensor_meta.storage;
                            if (tensor_val->tensor_meta.storage) {
                                tensor_val->tensor_meta.dtype = tensor_val->tensor_meta.storage->dtype_name;
                            }
                        }
                    }

                    ctx.push(std::move(tensor_val));
                    break;
                }

                if (callable->class_name == "_rebuild_from_type_v2") {
                    // Alternative tensor reconstruction
                    auto result = std::make_shared<PickleValue>();
                    result->type = PickleValue::Type::Reduce;
                    ctx.push(std::move(result));
                    break;
                }
            }

            // Generic reduce result
            auto result = std::make_shared<PickleValue>();
            result->type = PickleValue::Type::Reduce;
            ctx.push(std::move(result));
            break;
        }

        case NEWOBJ:
        case NEWOBJ_EX: {
            auto args = ctx.pop();
            auto cls = ctx.pop();

            // Handle storage objects
            if (cls->type == PickleValue::Type::Global &&
                (cls->class_name.find("Storage") != std::string::npos)) {

                auto storage_val = std::make_shared<PickleValue>();
                storage_val->type = PickleValue::Type::Reduce;
                storage_val->tensor_meta.storage = std::make_shared<TensorStorage>();

                // Extract dtype from class name
                std::string dtype_name = cls->class_name;
                size_t pos = dtype_name.find("Storage");
                if (pos != std::string::npos) {
                    storage_val->tensor_meta.storage->dtype_name = dtype_name.substr(0, pos);
                }

                ctx.push(std::move(storage_val));
                break;
            }

            auto result = std::make_shared<PickleValue>();
            result->type = PickleValue::Type::Reduce;
            ctx.push(std::move(result));
            break;
        }

        case BUILD: {
            auto state = ctx.pop();
            auto obj = ctx.top();

            // Handle storage state updates
            if (obj->type == PickleValue::Type::Reduce &&
                obj->tensor_meta.storage &&
                state->type == PickleValue::Type::Tuple) {

                // State typically contains (data, dtype_str, device, size)
                if (state->tuple_val.size() >= 1 &&
                    state->tuple_val[0]->type == PickleValue::Type::Bytes) {
                    obj->tensor_meta.storage->data = state->tuple_val[0]->bytes_val;
                }
            }
            break;
        }

        case PUT: {
            std::string line = read_line(ctx);
            // Security: wrap stoul in try-catch (HIGH-01 fix)
            size_t idx;
            try {
                idx = std::stoul(line);
            } catch (const std::exception& e) {
                throw ModelParseError("Invalid memo index in pickle: '" + line + "'", "PyTorch");
            }
            // Security: validate memo size limit
            if (ctx.memo.size() >= PickleLimits::MAX_MEMO_SIZE) {
                throw ModelParseError("Memo size limit exceeded", "PyTorch");
            }
            ctx.memo[idx] = ctx.top();
            break;
        }

        case BINPUT: {
            uint8_t idx = read_byte(ctx);
            // Security: validate memo size limit
            if (ctx.memo.size() >= PickleLimits::MAX_MEMO_SIZE) {
                throw ModelParseError("Memo size limit exceeded", "PyTorch");
            }
            ctx.memo[idx] = ctx.top();
            break;
        }

        case LONG_BINPUT: {
            uint32_t idx = read_uint32(ctx);
            // Security: validate memo size limit
            if (ctx.memo.size() >= PickleLimits::MAX_MEMO_SIZE) {
                throw ModelParseError("Memo size limit exceeded", "PyTorch");
            }
            ctx.memo[idx] = ctx.top();
            break;
        }

        case MEMOIZE: {
            // Security: validate memo size limit
            if (ctx.memo.size() >= PickleLimits::MAX_MEMO_SIZE) {
                throw ModelParseError("Memo size limit exceeded", "PyTorch");
            }
            ctx.memo[ctx.memo.size()] = ctx.top();
            break;
        }

        case GET: {
            std::string line = read_line(ctx);
            // Security: wrap stoul in try-catch (HIGH-01 fix)
            size_t idx;
            try {
                idx = std::stoul(line);
            } catch (const std::exception& e) {
                throw ModelParseError("Invalid memo index in pickle: '" + line + "'", "PyTorch");
            }
            auto it = ctx.memo.find(idx);
            if (it != ctx.memo.end()) {
                ctx.push(it->second);
            } else {
                ctx.push(PickleValue::make_none());
            }
            break;
        }

        case BINGET: {
            uint8_t idx = read_byte(ctx);
            auto it = ctx.memo.find(idx);
            if (it != ctx.memo.end()) {
                ctx.push(it->second);
            } else {
                ctx.push(PickleValue::make_none());
            }
            break;
        }

        case LONG_BINGET: {
            uint32_t idx = read_uint32(ctx);
            auto it = ctx.memo.find(idx);
            if (it != ctx.memo.end()) {
                ctx.push(it->second);
            } else {
                ctx.push(PickleValue::make_none());
            }
            break;
        }

        case BINPERSID: {
            auto pid = ctx.pop();

            // Persistent ID for tensor storage
            if (pid->type == PickleValue::Type::Tuple &&
                pid->tuple_val.size() >= 5) {

                auto storage_val = std::make_shared<PickleValue>();
                storage_val->type = PickleValue::Type::PersId;
                storage_val->tensor_meta.storage = std::make_shared<TensorStorage>();

                // pid format: (storage_type, key, device, numel, ...)
                if (pid->tuple_val[0]->type == PickleValue::Type::String) {
                    std::string storage_type = pid->tuple_val[0]->string_val;
                    // Extract dtype from storage type (e.g., "FloatStorage" -> "Float")
                    size_t pos = storage_type.find("Storage");
                    if (pos != std::string::npos) {
                        storage_val->tensor_meta.storage->dtype_name = storage_type.substr(0, pos);
                    }
                }

                if (pid->tuple_val[1]->type == PickleValue::Type::String) {
                    storage_val->tensor_meta.storage->key = pid->tuple_val[1]->string_val;
                }

                if (pid->tuple_val[2]->type == PickleValue::Type::String) {
                    storage_val->tensor_meta.storage->location = pid->tuple_val[2]->string_val;
                }

                if (pid->tuple_val[3]->type == PickleValue::Type::Int) {
                    storage_val->tensor_meta.storage->size = static_cast<size_t>(pid->tuple_val[3]->int_val);
                }

                // Try to load storage data from ZIP archive
                std::string data_key = "data/" + storage_val->tensor_meta.storage->key;
                auto it = ctx.storage_data.find(data_key);
                if (it != ctx.storage_data.end()) {
                    storage_val->tensor_meta.storage->data = it->second;
                }

                ctx.push(std::move(storage_val));
                break;
            }

            ctx.push(PickleValue::make_none());
            break;
        }

        case POP: {
            if (!ctx.stack.empty()) {
                ctx.pop();
            }
            break;
        }

        case POP_MARK: {
            ctx.pop_mark();
            break;
        }

        case DUP: {
            ctx.push(ctx.top());
            break;
        }

        default: {
            // Unknown opcode - skip with warning
            ctx.warnings.push_back(
                "Unknown pickle opcode: " + std::to_string(opcode)
            );
            break;
        }
    }
}

// ============================================================================
// Main Parsing Functions
// ============================================================================

std::shared_ptr<PickleValue> PickleParser::parse_pickle(ParseContext& ctx) {
    while (!ctx.at_end()) {
        uint8_t opcode = read_byte(ctx);

        if (opcode == STOP) {
            break;
        }

        execute_opcode(ctx, opcode);
    }

    if (ctx.stack.empty()) {
        return PickleValue::make_none();
    }

    return ctx.stack.back();
}

PickleParser::ParseResult PickleParser::parse_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw ModelParseError("Cannot open file: " + path, "PyTorch");
    }

    auto size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> buffer(static_cast<size_t>(size));
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        throw ModelParseError("Failed to read file: " + path, "PyTorch");
    }

    return parse_buffer(buffer.data(), buffer.size());
}

PickleParser::ParseResult PickleParser::parse_buffer(const void* data, size_t size) {
    ParseResult result;

    // Check if it's a ZIP file (modern PyTorch checkpoints)
    if (is_zip_file(data, size)) {
        auto files = extract_zip(data, size);

        // Find the pickle file
        std::vector<uint8_t> pickle_data;
        for (const auto& [name, content] : files) {
            if (name.find("data.pkl") != std::string::npos ||
                name.find(".pkl") != std::string::npos) {
                pickle_data = content;
            }
        }

        if (pickle_data.empty()) {
            throw ModelParseError("No pickle data found in ZIP archive", "PyTorch");
        }

        ParseContext ctx(pickle_data.data(), pickle_data.size());
        ctx.storage_data = std::move(files);

        auto root = parse_pickle(ctx);
        result.warnings = std::move(ctx.warnings);

        // Extract state dict from parsed value
        if (root->type == PickleValue::Type::Dict) {
            for (const auto& [key, val] : root->dict_val) {
                if (val->type == PickleValue::Type::Tensor) {
                    result.state_dict[key] = val->tensor_meta;
                } else if (val->type == PickleValue::Type::Dict) {
                    // Nested module dict
                    for (const auto& [subkey, subval] : val->dict_val) {
                        if (subval->type == PickleValue::Type::Tensor) {
                            result.state_dict[key + "." + subkey] = subval->tensor_meta;
                        }
                    }
                }
            }
        }
    } else {
        // Raw pickle file (legacy format)
        ParseContext ctx(data, size);
        auto root = parse_pickle(ctx);
        result.warnings = std::move(ctx.warnings);

        if (root->type == PickleValue::Type::Dict) {
            for (const auto& [key, val] : root->dict_val) {
                if (val->type == PickleValue::Type::Tensor) {
                    result.state_dict[key] = val->tensor_meta;
                }
            }
        }
    }

    return result;
}

bool PickleParser::is_pytorch_checkpoint(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    uint8_t header[4];
    if (!file.read(reinterpret_cast<char*>(header), 4)) {
        return false;
    }

    // Check for ZIP header (modern checkpoints)
    if (header[0] == 'P' && header[1] == 'K' &&
        header[2] == 0x03 && header[3] == 0x04) {
        return true;
    }

    // Check for pickle header (legacy)
    // Protocol 2+ starts with 0x80
    if (header[0] == 0x80) {
        return true;
    }

    return false;
}

// ============================================================================
// Tensor Conversion
// ============================================================================

DType pytorch_dtype_to_dtype(const std::string& pytorch_dtype) {
    // PyTorch storage type names
    if (pytorch_dtype == "Float" || pytorch_dtype == "float") {
        return DType::Float32;
    }
    if (pytorch_dtype == "Double" || pytorch_dtype == "double") {
        return DType::Float64;
    }
    if (pytorch_dtype == "Half" || pytorch_dtype == "half") {
        return DType::Float16;
    }
    if (pytorch_dtype == "BFloat16" || pytorch_dtype == "bfloat16") {
        return DType::BFloat16;
    }
    if (pytorch_dtype == "Long" || pytorch_dtype == "long" || pytorch_dtype == "int64") {
        return DType::Int64;
    }
    if (pytorch_dtype == "Int" || pytorch_dtype == "int" || pytorch_dtype == "int32") {
        return DType::Int32;
    }
    if (pytorch_dtype == "Short" || pytorch_dtype == "short") {
        return DType::Int16;
    }
    if (pytorch_dtype == "Char" || pytorch_dtype == "char" || pytorch_dtype == "int8") {
        return DType::Int8;
    }
    if (pytorch_dtype == "Byte" || pytorch_dtype == "byte" || pytorch_dtype == "uint8") {
        return DType::UInt8;
    }
    if (pytorch_dtype == "Bool" || pytorch_dtype == "bool") {
        return DType::Bool;
    }

    // Default to float32
    return DType::Float32;
}

Tensor convert_pytorch_tensor(const TensorMeta& meta) {
    DType dtype = pytorch_dtype_to_dtype(meta.dtype);
    Tensor tensor(meta.shape, dtype);

    if (meta.storage && !meta.storage->data.empty()) {
        size_t elem_size = dtype_size(dtype);

        // Security: validate storage_offset is non-negative (MED-01 fix)
        if (meta.storage_offset < 0) {
            throw ModelParseError("Invalid negative storage offset in tensor", "PyTorch");
        }

        size_t offset_bytes = static_cast<size_t>(meta.storage_offset) * elem_size;
        size_t copy_size = tensor.size_bytes();
        size_t storage_size = meta.storage->data.size();

        // Security: check for overflow in offset calculation (MED-01 fix)
        if (meta.storage_offset > 0 &&
            static_cast<size_t>(meta.storage_offset) > SIZE_MAX / elem_size) {
            throw ModelParseError("Storage offset overflow in tensor", "PyTorch");
        }

        // Security: strict bounds check without overflow (MED-01 fix)
        if (offset_bytes > storage_size ||
            copy_size > storage_size - offset_bytes) {
            throw ModelParseError(
                "Tensor data out of bounds: offset=" + std::to_string(offset_bytes) +
                ", size=" + std::to_string(copy_size) +
                ", storage=" + std::to_string(storage_size), "PyTorch");
        }

        // Safe to copy now
        if (copy_size > 0) {
            std::memcpy(
                tensor.mutable_data_ptr(),
                meta.storage->data.data() + offset_bytes,
                copy_size
            );
        }
    }

    return tensor;
}

} // namespace pytorch
} // namespace import
} // namespace pyflame_rt
