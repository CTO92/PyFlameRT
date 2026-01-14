#include "pyflame_rt/import/onnx_importer.hpp"
#include "pyflame_rt/import/op_converter.hpp"
#include "pyflame_rt/import/shape_inference.hpp"
#include "onnx_parser.hpp"

#include <algorithm>
#include <cstring>
#include <unordered_set>

namespace pyflame_rt {
namespace import {

// ============================================================================
// ONNX to PyFlameRT Type Conversion
// ============================================================================

namespace {

/// Convert ONNX data type to PyFlameRT DType
DType convert_dtype(onnx_proto::TensorDataType onnx_type) {
    switch (onnx_type) {
        case onnx_proto::TensorDataType::FLOAT:
            return DType::Float32;
        case onnx_proto::TensorDataType::FLOAT16:
            return DType::Float16;
        case onnx_proto::TensorDataType::BFLOAT16:
            return DType::BFloat16;
        case onnx_proto::TensorDataType::DOUBLE:
            return DType::Float64;
        case onnx_proto::TensorDataType::INT64:
            return DType::Int64;
        case onnx_proto::TensorDataType::INT32:
            return DType::Int32;
        case onnx_proto::TensorDataType::INT16:
            return DType::Int16;
        case onnx_proto::TensorDataType::INT8:
            return DType::Int8;
        case onnx_proto::TensorDataType::UINT8:
            return DType::UInt8;
        case onnx_proto::TensorDataType::BOOL:
            return DType::Bool;
        default:
            throw OperatorConversionError(
                "dtype",
                "onnx",
                "Unsupported ONNX data type: " + std::to_string(static_cast<int>(onnx_type))
            );
    }
}

/// Convert ONNX TensorProto to PyFlameRT Tensor
Tensor convert_tensor(const onnx_proto::TensorProto& onnx_tensor) {
    DType dtype = convert_dtype(onnx_tensor.data_type);
    std::vector<int64_t> shape(onnx_tensor.dims.begin(), onnx_tensor.dims.end());

    Tensor tensor(shape, dtype);
    size_t expected_bytes = tensor.size_bytes();

    // Get data from raw_data or typed arrays
    if (!onnx_tensor.raw_data.empty()) {
        if (onnx_tensor.raw_data.size() != expected_bytes) {
            throw ModelParseError(
                "Tensor '" + onnx_tensor.name + "' has mismatched data size: expected " +
                std::to_string(expected_bytes) + " bytes, got " +
                std::to_string(onnx_tensor.raw_data.size()),
                "ONNX"
            );
        }
        std::memcpy(tensor.mutable_data_ptr(), onnx_tensor.raw_data.data(), expected_bytes);
    } else if (!onnx_tensor.float_data.empty()) {
        if (dtype != DType::Float32) {
            throw ModelParseError(
                "Tensor '" + onnx_tensor.name + "' has float_data but dtype is not Float32",
                "ONNX"
            );
        }
        std::memcpy(tensor.mutable_data_ptr(), onnx_tensor.float_data.data(), expected_bytes);
    } else if (!onnx_tensor.int64_data.empty()) {
        if (dtype != DType::Int64) {
            throw ModelParseError(
                "Tensor '" + onnx_tensor.name + "' has int64_data but dtype is not Int64",
                "ONNX"
            );
        }
        std::memcpy(tensor.mutable_data_ptr(), onnx_tensor.int64_data.data(), expected_bytes);
    } else if (!onnx_tensor.int32_data.empty()) {
        if (dtype != DType::Int32) {
            throw ModelParseError(
                "Tensor '" + onnx_tensor.name + "' has int32_data but dtype is not Int32",
                "ONNX"
            );
        }
        std::memcpy(tensor.mutable_data_ptr(), onnx_tensor.int32_data.data(), expected_bytes);
    } else if (!onnx_tensor.double_data.empty()) {
        if (dtype != DType::Float64) {
            throw ModelParseError(
                "Tensor '" + onnx_tensor.name + "' has double_data but dtype is not Float64",
                "ONNX"
            );
        }
        std::memcpy(tensor.mutable_data_ptr(), onnx_tensor.double_data.data(), expected_bytes);
    }
    // Empty tensor is allowed (will have zeros)

    return tensor;
}

/// Convert ONNX ValueInfoProto to PyFlameRT TensorInfo
TensorInfo convert_value_info(const onnx_proto::ValueInfoProto& value_info) {
    TensorInfo info;
    info.name = value_info.name;

    const auto& tensor_type = value_info.type.tensor_type;

    // Convert dtype
    if (tensor_type.elem_type != onnx_proto::TensorDataType::UNDEFINED) {
        info.dtype = convert_dtype(tensor_type.elem_type);
    } else {
        info.dtype = DType::Float32;  // Default
    }

    // Convert shape
    for (const auto& dim : tensor_type.shape.dims) {
        if (dim.is_concrete()) {
            info.shape.push_back(dim.dim_value.value());
        } else {
            info.shape.push_back(std::nullopt);  // Dynamic dimension
        }
    }

    return info;
}

/// Convert ONNX attribute to std::any for OpConversionContext
std::any convert_attribute(const onnx_proto::AttributeProto& attr) {
    switch (attr.type) {
        case onnx_proto::AttributeType::FLOAT:
            return attr.f;
        case onnx_proto::AttributeType::INT:
            return attr.i;
        case onnx_proto::AttributeType::STRING:
            return attr.s;
        case onnx_proto::AttributeType::FLOATS:
            return attr.floats;
        case onnx_proto::AttributeType::INTS:
            return attr.ints;
        case onnx_proto::AttributeType::STRINGS:
            return attr.strings;
        default:
            // Unsupported attribute types return empty any
            return std::any{};
    }
}

} // anonymous namespace

// ============================================================================
// ONNXImporter Implementation
// ============================================================================

class ONNXImporter::Impl {
public:
    Impl() = default;

    ImportResult import_file(const std::string& path, const ImportOptions& options) {
        try {
            auto model = ONNXParser::parse_file(path);
            return import_model_proto(model, options);
        } catch (const ModelParseError&) {
            throw;
        } catch (const std::exception& e) {
            throw ModelParseError(e.what(), "ONNX");
        }
    }

    ImportResult import_buffer(const void* data, size_t size, const ImportOptions& options) {
        try {
            auto model = ONNXParser::parse_buffer(data, size);
            return import_model_proto(model, options);
        } catch (const ModelParseError&) {
            throw;
        } catch (const std::exception& e) {
            throw ModelParseError(e.what(), "ONNX");
        }
    }

private:
    ImportResult import_model_proto(const onnx_proto::ModelProto& model,
                                    const ImportOptions& options) {
        ImportResult result;
        result.graph = std::make_unique<Graph>(model.graph.name);

        // Extract metadata
        result.metadata.producer_name = model.producer_name;
        result.metadata.producer_version = model.producer_version;
        result.metadata.domain = model.domain;
        result.metadata.graph_name = model.graph.name;
        result.metadata.version = model.model_version;
        result.metadata.description = model.doc_string;

        // Get opset version
        int opset_version = options.opset_version;
        if (opset_version == 0) {
            opset_version = static_cast<int>(model.get_opset_version());
            if (opset_version == 0) {
                opset_version = 9;  // Default to opset 9
            }
        }

        // Validate opset version
        if (!ONNXImporter::is_opset_supported(opset_version)) {
            result.warnings.push_back(
                "ONNX opset version " + std::to_string(opset_version) +
                " is outside supported range [" +
                std::to_string(ONNXImporter::MIN_OPSET_VERSION) + ", " +
                std::to_string(ONNXImporter::MAX_OPSET_VERSION) + "]"
            );
        }

        // Build set of initializer names for quick lookup
        std::unordered_set<std::string> initializer_names;
        for (const auto& init : model.graph.initializer) {
            initializer_names.insert(init.name);
        }

        // Add graph inputs (excluding initializers)
        for (const auto& input : model.graph.input) {
            // Skip inputs that are actually initializers
            if (initializer_names.count(input.name)) {
                continue;
            }
            TensorInfo info = convert_value_info(input);

            // Override shape if provided in options
            auto it = options.input_shapes.find(input.name);
            if (it != options.input_shapes.end()) {
                info.shape = to_shape(it->second);
            }

            result.graph->add_input(info);
        }

        // Add graph outputs
        for (const auto& output : model.graph.output) {
            result.graph->add_output(convert_value_info(output));
        }

        // Load initializers (weights)
        if (options.load_initializers) {
            for (const auto& init : model.graph.initializer) {
                try {
                    Tensor tensor = convert_tensor(init);
                    result.graph->add_initializer(init.name, std::move(tensor));
                    result.stats.total_initializers++;
                    result.stats.total_weight_bytes += init.size_bytes();
                } catch (const std::exception& e) {
                    result.warnings.push_back(
                        "Failed to load initializer '" + init.name + "': " + e.what()
                    );
                }
            }
        }

        // Convert nodes
        auto& converter_registry = OpConverterRegistry::instance();

        for (const auto& onnx_node : model.graph.node) {
            result.stats.total_ops++;

            // Build conversion context
            OpConversionContext ctx;
            ctx.source_op = onnx_node.op_type;
            ctx.framework = "onnx";
            ctx.opset_version = opset_version;
            ctx.inputs = onnx_node.input;
            ctx.outputs = onnx_node.output;
            ctx.node_name = onnx_node.name;
            ctx.graph = result.graph.get();

            // Convert attributes
            for (const auto& attr : onnx_node.attribute) {
                ctx.attrs[attr.name] = convert_attribute(attr);
            }

            // Try to convert using registered converter
            auto converter = converter_registry.get("onnx", onnx_node.op_type);

            if (converter) {
                auto conv_result = converter(ctx);

                if (conv_result.success) {
                    for (auto& node : conv_result.nodes) {
                        result.graph->add_node(std::move(node));
                    }
                    for (const auto& warning : conv_result.warnings) {
                        result.warnings.push_back(warning);
                    }
                    result.stats.mapped_ops++;
                } else {
                    if (options.allow_unsupported) {
                        // Create placeholder node
                        auto placeholder = std::make_shared<Node>(
                            onnx_node.name.empty() ? onnx_node.op_type + "_placeholder" : onnx_node.name,
                            "Placeholder_" + onnx_node.op_type,
                            onnx_node.input,
                            onnx_node.output
                        );
                        result.graph->add_node(std::move(placeholder));
                        result.stats.unsupported_ops++;
                        result.stats.unsupported_op_types.push_back(onnx_node.op_type);
                        result.warnings.push_back(
                            "Operator '" + onnx_node.op_type + "' conversion failed: " +
                            conv_result.error + " (created placeholder)"
                        );
                    } else {
                        throw OperatorConversionError(
                            onnx_node.op_type,
                            "onnx",
                            conv_result.error
                        );
                    }
                }
            } else {
                // No converter registered
                if (options.allow_unsupported) {
                    // Create placeholder node
                    auto placeholder = std::make_shared<Node>(
                        onnx_node.name.empty() ? onnx_node.op_type + "_placeholder" : onnx_node.name,
                        "Placeholder_" + onnx_node.op_type,
                        onnx_node.input,
                        onnx_node.output
                    );
                    result.graph->add_node(std::move(placeholder));
                    result.stats.unsupported_ops++;
                    result.stats.unsupported_op_types.push_back(onnx_node.op_type);
                    result.warnings.push_back(
                        "No converter for ONNX operator '" + onnx_node.op_type +
                        "' (created placeholder)"
                    );
                } else {
                    throw OperatorConversionError(
                        onnx_node.op_type,
                        "onnx",
                        "No converter registered"
                    );
                }
            }
        }

        // Run shape inference if requested
        if (options.infer_shapes) {
            ShapeInference shape_inference(*result.graph);

            // Set input shapes from options
            for (const auto& [name, shape] : options.input_shapes) {
                shape_inference.set_input_shape(name, shape);
            }

            auto shape_result = shape_inference.infer();
            if (!shape_result.complete) {
                for (const auto& error : shape_result.errors) {
                    result.warnings.push_back("Shape inference: " + error);
                }
            }
        }

        return result;
    }
};

// ============================================================================
// ONNXImporter Public API
// ============================================================================

ONNXImporter::ONNXImporter() : impl_(std::make_unique<Impl>()) {}

ONNXImporter::~ONNXImporter() = default;

ONNXImporter::ONNXImporter(ONNXImporter&&) noexcept = default;
ONNXImporter& ONNXImporter::operator=(ONNXImporter&&) noexcept = default;

ImportResult ONNXImporter::import_file(const std::string& path, const ImportOptions& options) {
    return impl_->import_file(path, options);
}

ImportResult ONNXImporter::import_buffer(const void* data, size_t size, const ImportOptions& options) {
    return impl_->import_buffer(data, size, options);
}

int ONNXImporter::get_opset_version(const std::string& path) {
    return ONNXParser::get_opset_version(path);
}

// ============================================================================
// ONNX Importer Registration
// ============================================================================

void register_onnx_importer() {
    static bool registered = false;
    if (!registered) {
        ImporterRegistry::instance().register_importer(
            std::make_unique<ONNXImporter>()
        );
        registered = true;
    }
}

// Auto-register at static initialization
namespace {
    struct ONNXImporterRegistrar {
        ONNXImporterRegistrar() {
            register_onnx_importer();
        }
    };
    static ONNXImporterRegistrar onnx_importer_registrar;
}

} // namespace import
} // namespace pyflame_rt
