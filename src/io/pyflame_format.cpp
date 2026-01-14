#include "pyflame_format.hpp"
#include "pyflame_rt/errors.hpp"
#include "pyflame_rt/types.hpp"
#include <fstream>
#include <cstring>

namespace pyflame_rt {

namespace {

// ============================================================================
// Stream Safety Helpers (CRIT-03 fix)
// ============================================================================

/// Check stream state and throw on failure
inline void check_stream(std::istream& is, const char* context) {
    if (is.fail()) {
        throw InvalidModelError(std::string("Stream read failed: ") + context);
    }
}

/// Read exactly N bytes and validate (CRIT-03 fix)
inline void read_exact(std::istream& is, void* buffer, size_t size, const char* context) {
    is.read(static_cast<char*>(buffer), static_cast<std::streamsize>(size));

    // CRIT-03 fix: Check multiple error conditions
    // 1. Check fail bit (includes badbit)
    // 2. Check that gcount is non-negative and equals expected size
    //    (gcount returns std::streamsize, which is signed)
    std::streamsize bytes_read = is.gcount();
    if (is.fail() || bytes_read < 0 ||
        static_cast<size_t>(bytes_read) != size) {
        throw InvalidModelError(
            std::string("Incomplete read: ") + context +
            " (expected " + std::to_string(size) + " bytes, got " +
            std::to_string(bytes_read) + ")");
    }
}

// ============================================================================
// Binary Serialization Helpers (Write - unchanged)
// ============================================================================

void write_uint32(std::ostream& os, uint32_t value) {
    os.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void write_uint64(std::ostream& os, uint64_t value) {
    os.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void write_int64(std::ostream& os, int64_t value) {
    os.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void write_string(std::ostream& os, const std::string& str) {
    write_uint64(os, str.size());
    os.write(str.data(), str.size());
}

void write_bytes(std::ostream& os, const void* data, size_t size) {
    write_uint64(os, size);
    os.write(static_cast<const char*>(data), size);
}

// ============================================================================
// Binary Deserialization Helpers (Read - with security fixes)
// ============================================================================

uint32_t read_uint32(std::istream& is) {
    uint32_t value = 0;
    read_exact(is, &value, sizeof(value), "uint32");
    return value;
}

uint64_t read_uint64(std::istream& is) {
    uint64_t value = 0;
    read_exact(is, &value, sizeof(value), "uint64");
    return value;
}

int64_t read_int64(std::istream& is) {
    int64_t value = 0;
    read_exact(is, &value, sizeof(value), "int64");
    return value;
}

/// Read string with size limit (CRIT-01 fix)
std::string read_string(std::istream& is) {
    uint64_t size = read_uint64(is);

    // Security check: limit string size
    if (size > DeserializationLimits::MAX_STRING_SIZE) {
        throw InvalidModelError(
            "String size exceeds limit: " + std::to_string(size) +
            " > " + std::to_string(DeserializationLimits::MAX_STRING_SIZE));
    }

    std::string str(static_cast<size_t>(size), '\0');
    if (size > 0) {
        read_exact(is, &str[0], static_cast<size_t>(size), "string data");
    }
    return str;
}

/// Read bytes with size limit (CRIT-01 fix)
std::vector<char> read_bytes(std::istream& is, uint64_t max_size = DeserializationLimits::MAX_TENSOR_DATA_SIZE) {
    uint64_t size = read_uint64(is);

    // Security check: limit data size
    if (size > max_size) {
        throw InvalidModelError(
            "Data size exceeds limit: " + std::to_string(size) +
            " > " + std::to_string(max_size));
    }

    std::vector<char> data(static_cast<size_t>(size));
    if (size > 0) {
        read_exact(is, data.data(), static_cast<size_t>(size), "byte data");
    }
    return data;
}

void write_tensor_info(std::ostream& os, const TensorInfo& info) {
    write_string(os, info.name);
    write_uint32(os, static_cast<uint32_t>(info.dtype));
    write_uint64(os, info.shape.size());
    for (const auto& dim : info.shape) {
        int64_t val = dim.has_value() ? dim.value() : -1;
        write_int64(os, val);
    }
}

/// Read tensor info with validation (CRIT-04 fix)
TensorInfo read_tensor_info(std::istream& is) {
    TensorInfo info;
    info.name = read_string(is);

    // Security check: validate dtype value before casting (CRIT-04)
    uint32_t dtype_val = read_uint32(is);
    if (!validate_dtype_value(dtype_val)) {
        throw InvalidModelError(
            "Invalid dtype value: " + std::to_string(dtype_val));
    }
    info.dtype = static_cast<DType>(dtype_val);

    uint64_t ndim = read_uint64(is);

    // Security check: limit number of dimensions
    if (ndim > DeserializationLimits::MAX_DIMS) {
        throw InvalidModelError(
            "Too many dimensions: " + std::to_string(ndim) +
            " > " + std::to_string(DeserializationLimits::MAX_DIMS));
    }

    info.shape.reserve(static_cast<size_t>(ndim));
    for (uint64_t i = 0; i < ndim; ++i) {
        int64_t val = read_int64(is);
        if (val < 0) {
            info.shape.push_back(std::nullopt);
        } else {
            info.shape.push_back(val);
        }
    }
    return info;
}

void write_tensor(std::ostream& os, const std::string& name, const Tensor& tensor) {
    write_string(os, name);
    write_uint32(os, static_cast<uint32_t>(tensor.dtype()));
    write_uint64(os, tensor.ndim());
    for (size_t i = 0; i < tensor.ndim(); ++i) {
        write_int64(os, tensor.shape()[i]);
    }
    write_bytes(os, tensor.data(), tensor.size_bytes());
}

/// Read tensor with comprehensive validation (CRIT-02, CRIT-04 fixes)
std::pair<std::string, Tensor> read_tensor(std::istream& is) {
    std::string name = read_string(is);

    // Security check: validate dtype value (CRIT-04)
    uint32_t dtype_val = read_uint32(is);
    if (!validate_dtype_value(dtype_val)) {
        throw InvalidModelError(
            "Invalid dtype value in tensor '" + name + "': " + std::to_string(dtype_val));
    }
    DType dtype = static_cast<DType>(dtype_val);

    uint64_t ndim = read_uint64(is);

    // Security check: limit dimensions
    if (ndim > DeserializationLimits::MAX_DIMS) {
        throw InvalidModelError(
            "Too many dimensions in tensor '" + name + "': " + std::to_string(ndim));
    }

    std::vector<int64_t> shape(static_cast<size_t>(ndim));
    for (uint64_t i = 0; i < ndim; ++i) {
        shape[i] = read_int64(is);
        // Security check: validate non-negative dimensions
        if (shape[i] < 0) {
            throw InvalidModelError(
                "Negative dimension in tensor '" + name + "' at axis " + std::to_string(i));
        }
        // CRIT-02 fix: validate dimension is within reasonable bounds
        if (shape[i] > DeserializationLimits::MAX_DIMENSION_SIZE) {
            throw InvalidModelError(
                "Dimension too large in tensor '" + name + "' at axis " + std::to_string(i) +
                ": " + std::to_string(shape[i]) + " > " +
                std::to_string(DeserializationLimits::MAX_DIMENSION_SIZE));
        }
    }

    // Create tensor (this will use checked_product internally after CRIT-05 fix)
    Tensor tensor(shape, dtype);

    // Read tensor data with size limit
    auto data = read_bytes(is, DeserializationLimits::MAX_TENSOR_DATA_SIZE);

    // Security check: validate data size matches tensor size (CRIT-02)
    size_t expected_size = tensor.size_bytes();
    if (data.size() != expected_size) {
        throw InvalidModelError(
            "Data size mismatch for tensor '" + name + "': got " +
            std::to_string(data.size()) + ", expected " + std::to_string(expected_size));
    }

    // Safe to copy now that sizes are verified
    if (expected_size > 0) {
        std::memcpy(tensor.data(), data.data(), expected_size);
    }

    return {name, std::move(tensor)};
}

void write_node(std::ostream& os, const Node& node) {
    write_string(os, node.name());
    write_string(os, node.op_type());

    // Inputs
    write_uint64(os, node.inputs().size());
    for (const auto& inp : node.inputs()) {
        write_string(os, inp);
    }

    // Outputs
    write_uint64(os, node.outputs().size());
    for (const auto& out : node.outputs()) {
        write_string(os, out);
    }

    // Attributes (simplified - only int64 and float for now)
    const auto& attrs = node.attributes();
    write_uint64(os, attrs.size());
    for (const auto& [name, value] : attrs) {
        write_string(os, name);
        if (std::holds_alternative<int64_t>(value)) {
            write_uint32(os, 0);  // Type: int64
            write_int64(os, std::get<int64_t>(value));
        } else if (std::holds_alternative<float>(value)) {
            write_uint32(os, 1);  // Type: float
            float f = std::get<float>(value);
            os.write(reinterpret_cast<const char*>(&f), sizeof(f));
        } else if (std::holds_alternative<std::string>(value)) {
            write_uint32(os, 2);  // Type: string
            write_string(os, std::get<std::string>(value));
        } else if (std::holds_alternative<std::vector<int64_t>>(value)) {
            write_uint32(os, 3);  // Type: int64 vector
            const auto& vec = std::get<std::vector<int64_t>>(value);
            write_uint64(os, vec.size());
            for (auto v : vec) {
                write_int64(os, v);
            }
        } else if (std::holds_alternative<std::vector<float>>(value)) {
            write_uint32(os, 4);  // Type: float vector
            const auto& vec = std::get<std::vector<float>>(value);
            write_uint64(os, vec.size());
            for (float v : vec) {
                os.write(reinterpret_cast<const char*>(&v), sizeof(v));
            }
        } else {
            write_uint32(os, 255);  // Unknown type
        }
    }
}

/// Read node with validation (CRIT-01, MED-02 fixes)
std::shared_ptr<Node> read_node(std::istream& is) {
    std::string name = read_string(is);
    std::string op_type = read_string(is);

    // Inputs with count limit
    uint64_t num_inputs = read_uint64(is);
    if (num_inputs > DeserializationLimits::MAX_ARRAY_COUNT) {
        throw InvalidModelError(
            "Too many inputs for node '" + name + "': " + std::to_string(num_inputs));
    }
    std::vector<std::string> inputs;
    inputs.reserve(static_cast<size_t>(num_inputs));
    for (uint64_t i = 0; i < num_inputs; ++i) {
        inputs.push_back(read_string(is));
    }

    // Outputs with count limit
    uint64_t num_outputs = read_uint64(is);
    if (num_outputs > DeserializationLimits::MAX_ARRAY_COUNT) {
        throw InvalidModelError(
            "Too many outputs for node '" + name + "': " + std::to_string(num_outputs));
    }
    std::vector<std::string> outputs;
    outputs.reserve(static_cast<size_t>(num_outputs));
    for (uint64_t i = 0; i < num_outputs; ++i) {
        outputs.push_back(read_string(is));
    }

    auto node = std::make_shared<Node>(name, op_type, inputs, outputs);

    // Attributes with count limit
    uint64_t num_attrs = read_uint64(is);
    if (num_attrs > DeserializationLimits::MAX_ATTRIBUTES) {
        throw InvalidModelError(
            "Too many attributes for node '" + name + "': " + std::to_string(num_attrs));
    }

    for (uint64_t i = 0; i < num_attrs; ++i) {
        std::string attr_name = read_string(is);
        uint32_t attr_type = read_uint32(is);

        if (attr_type == 0) {  // int64
            node->set_attr(attr_name, read_int64(is));
        } else if (attr_type == 1) {  // float
            float f = 0.0f;
            read_exact(is, &f, sizeof(f), "float attribute");
            node->set_attr(attr_name, f);
        } else if (attr_type == 2) {  // string
            node->set_attr(attr_name, read_string(is));
        } else if (attr_type == 3) {  // int64 vector
            uint64_t vec_size = read_uint64(is);
            if (vec_size > DeserializationLimits::MAX_ARRAY_COUNT) {
                throw InvalidModelError(
                    "Vector attribute '" + attr_name + "' too large: " + std::to_string(vec_size));
            }
            std::vector<int64_t> vec(static_cast<size_t>(vec_size));
            for (uint64_t j = 0; j < vec_size; ++j) {
                vec[j] = read_int64(is);
            }
            node->set_attr(attr_name, vec);
        } else if (attr_type == 4) {  // float vector
            uint64_t vec_size = read_uint64(is);
            if (vec_size > DeserializationLimits::MAX_ARRAY_COUNT) {
                throw InvalidModelError(
                    "Vector attribute '" + attr_name + "' too large: " + std::to_string(vec_size));
            }
            std::vector<float> vec(static_cast<size_t>(vec_size));
            for (uint64_t j = 0; j < vec_size; ++j) {
                read_exact(is, &vec[j], sizeof(float), "float vector element");
            }
            node->set_attr(attr_name, vec);
        } else if (attr_type == 255) {
            // Skip unknown marker (written for unsupported types during save)
            // No data follows type 255
        } else {
            // MED-02 fix: reject unknown attribute types instead of silent skip
            throw InvalidModelError(
                "Unknown attribute type " + std::to_string(attr_type) +
                " for attribute '" + attr_name + "' in node '" + name + "'");
        }
    }

    return node;
}

} // anonymous namespace

std::unique_ptr<Graph> PyFlameFormat::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw InvalidModelError("Cannot open file", path);
    }

    // Read and validate magic
    char magic[4];
    read_exact(file, magic, 4, "magic bytes");
    if (std::memcmp(magic, MAGIC, 4) != 0) {
        throw InvalidModelError("Invalid magic bytes", path);
    }

    // Read and validate version
    uint32_t version = read_uint32(file);
    if (version > VERSION) {
        throw InvalidModelError("Unsupported version: " + std::to_string(version), path);
    }

    // Read graph name
    std::string graph_name = read_string(file);
    auto graph = std::make_unique<Graph>(graph_name);

    // Read inputs with count limit
    uint64_t num_inputs = read_uint64(file);
    if (num_inputs > DeserializationLimits::MAX_ARRAY_COUNT) {
        throw InvalidModelError(
            "Too many graph inputs: " + std::to_string(num_inputs), path);
    }
    for (uint64_t i = 0; i < num_inputs; ++i) {
        graph->add_input(read_tensor_info(file));
    }

    // Read outputs with count limit
    uint64_t num_outputs = read_uint64(file);
    if (num_outputs > DeserializationLimits::MAX_ARRAY_COUNT) {
        throw InvalidModelError(
            "Too many graph outputs: " + std::to_string(num_outputs), path);
    }
    for (uint64_t i = 0; i < num_outputs; ++i) {
        graph->add_output(read_tensor_info(file));
    }

    // Read initializers with count limit
    uint64_t num_initializers = read_uint64(file);
    if (num_initializers > DeserializationLimits::MAX_ARRAY_COUNT) {
        throw InvalidModelError(
            "Too many initializers: " + std::to_string(num_initializers), path);
    }
    for (uint64_t i = 0; i < num_initializers; ++i) {
        auto [name, tensor] = read_tensor(file);
        graph->add_initializer(name, std::move(tensor));
    }

    // Read nodes with count limit
    uint64_t num_nodes = read_uint64(file);
    if (num_nodes > DeserializationLimits::MAX_NODES) {
        throw InvalidModelError(
            "Too many nodes: " + std::to_string(num_nodes), path);
    }
    for (uint64_t i = 0; i < num_nodes; ++i) {
        graph->add_node(read_node(file));
    }

    // Verify we consumed the expected content (optional integrity check)
    check_stream(file, "end of file");

    return graph;
}

void PyFlameFormat::save(const Graph& graph, const std::string& path) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw InvalidModelError("Cannot create file", path);
    }

    // Write magic and version
    file.write(MAGIC, 4);
    write_uint32(file, VERSION);

    // Write graph name
    write_string(file, graph.name());

    // Write inputs
    write_uint64(file, graph.inputs().size());
    for (const auto& info : graph.inputs()) {
        write_tensor_info(file, info);
    }

    // Write outputs
    write_uint64(file, graph.outputs().size());
    for (const auto& info : graph.outputs()) {
        write_tensor_info(file, info);
    }

    // Write initializers
    write_uint64(file, graph.initializers().size());
    for (const auto& [name, tensor] : graph.initializers()) {
        write_tensor(file, name, tensor);
    }

    // Write nodes
    write_uint64(file, graph.num_nodes());
    for (const auto& node : graph.nodes()) {
        write_node(file, *node);
    }
}

} // namespace pyflame_rt
