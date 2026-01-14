#include "pyflame_rt/import/torchscript_importer.hpp"
#include "pyflame_rt/import/op_converter.hpp"
#include "pyflame_rt/import/shape_inference.hpp"
#include "../pytorch/pickle_parser.hpp"

#include <fstream>
#include <cstring>
#include <unordered_set>

namespace pyflame_rt {
namespace import {

// ============================================================================
// TorchScript Archive Structure
// ============================================================================

namespace {

/// Structure representing a TorchScript archive
struct TorchScriptArchive {
    std::vector<uint8_t> code;           // bytecode.pkl or code/
    std::vector<uint8_t> model;          // model.pkl (graph definition)
    std::vector<uint8_t> data;           // data.pkl (tensor metadata)
    std::unordered_map<std::string, std::vector<uint8_t>> constants;
    std::unordered_map<std::string, std::vector<uint8_t>> tensors;

    std::string version;
    bool is_torchscript = false;
};

/// Check if a ZIP file is a TorchScript archive
bool check_torchscript_archive(const std::unordered_map<std::string, std::vector<uint8_t>>& files) {
    // TorchScript archives contain either:
    // - version (required)
    // - code/ or bytecode.pkl
    // - data.pkl or data/ directory

    for (const auto& [name, content] : files) {
        if (name.find("version") != std::string::npos ||
            name.find("code/") != std::string::npos ||
            name.find("bytecode.pkl") != std::string::npos) {
            return true;
        }
    }
    return false;
}

/// Parse the TorchScript archive structure
TorchScriptArchive parse_archive(const void* data, size_t size) {
    TorchScriptArchive archive;

    // Check if it's a ZIP file
    if (size < 4) {
        return archive;
    }

    const uint8_t* bytes = static_cast<const uint8_t*>(data);
    if (bytes[0] != 'P' || bytes[1] != 'K' || bytes[2] != 0x03 || bytes[3] != 0x04) {
        return archive;
    }

    // Extract ZIP contents
    auto files = pytorch::PickleParser::extract_zip(data, size);

    archive.is_torchscript = check_torchscript_archive(files);
    if (!archive.is_torchscript) {
        return archive;
    }

    // Extract specific files
    for (const auto& [name, content] : files) {
        if (name.find("version") != std::string::npos) {
            archive.version = std::string(content.begin(), content.end());
            // Remove newlines
            while (!archive.version.empty() &&
                   (archive.version.back() == '\n' || archive.version.back() == '\r')) {
                archive.version.pop_back();
            }
        } else if (name.find("model.pkl") != std::string::npos) {
            archive.model = content;
        } else if (name.find("data.pkl") != std::string::npos ||
                   name.find("archive/data.pkl") != std::string::npos) {
            archive.data = content;
        } else if (name.find("constants.pkl") != std::string::npos) {
            archive.constants[name] = content;
        } else if (name.find("data/") != std::string::npos) {
            archive.tensors[name] = content;
        } else if (name.find("bytecode.pkl") != std::string::npos) {
            archive.code = content;
        }
    }

    return archive;
}

/// Convert TorchScript operator name to PyFlameRT operator
std::string convert_ts_op_name(const std::string& ts_op) {
    // TorchScript uses aten:: namespace
    static const std::unordered_map<std::string, std::string> op_map = {
        // Math
        {"aten::add", "Add"},
        {"aten::sub", "Sub"},
        {"aten::mul", "Mul"},
        {"aten::div", "Div"},
        {"aten::neg", "Neg"},
        {"aten::abs", "Abs"},
        {"aten::sqrt", "Sqrt"},
        {"aten::exp", "Exp"},
        {"aten::log", "Log"},
        {"aten::pow", "Pow"},
        {"aten::matmul", "MatMul"},
        {"aten::mm", "MatMul"},
        {"aten::bmm", "MatMul"},
        {"aten::linear", "Gemm"},
        {"aten::addmm", "Gemm"},

        // Activations
        {"aten::relu", "Relu"},
        {"aten::relu_", "Relu"},
        {"aten::sigmoid", "Sigmoid"},
        {"aten::tanh", "Tanh"},
        {"aten::softmax", "Softmax"},
        {"aten::leaky_relu", "LeakyRelu"},
        {"aten::leaky_relu_", "LeakyRelu"},
        {"aten::elu", "Elu"},
        {"aten::selu", "Selu"},
        {"aten::gelu", "Gelu"},

        // Tensor operations
        {"aten::view", "Reshape"},
        {"aten::reshape", "Reshape"},
        {"aten::transpose", "Transpose"},
        {"aten::permute", "Transpose"},
        {"aten::cat", "Concat"},
        {"aten::split", "Split"},
        {"aten::squeeze", "Squeeze"},
        {"aten::unsqueeze", "Unsqueeze"},
        {"aten::flatten", "Flatten"},
        {"aten::slice", "Slice"},
        {"aten::select", "Slice"},
        {"aten::gather", "Gather"},
        {"aten::expand", "Expand"},

        // Reductions
        {"aten::sum", "ReduceSum"},
        {"aten::mean", "ReduceMean"},
        {"aten::max", "ReduceMax"},
        {"aten::min", "ReduceMin"},
        {"aten::prod", "ReduceProd"},
        {"aten::argmax", "ArgMax"},
        {"aten::argmin", "ArgMin"},

        // NN layers
        {"aten::conv2d", "Conv"},
        {"aten::conv1d", "Conv"},
        {"aten::_convolution", "Conv"},
        {"aten::max_pool2d", "MaxPool"},
        {"aten::avg_pool2d", "AveragePool"},
        {"aten::adaptive_avg_pool2d", "GlobalAveragePool"},
        {"aten::batch_norm", "BatchNorm"},
        {"aten::layer_norm", "LayerNorm"},
        {"aten::dropout", "Dropout"},

        // Comparison
        {"aten::eq", "Equal"},
        {"aten::gt", "Greater"},
        {"aten::lt", "Less"},
        {"aten::ge", "GreaterOrEqual"},
        {"aten::le", "LessOrEqual"},

        // Other
        {"aten::contiguous", "Identity"},
        {"aten::clone", "Identity"},
        {"aten::detach", "Identity"},
        {"aten::to", "Cast"},
        {"aten::type_as", "Cast"},
    };

    auto it = op_map.find(ts_op);
    if (it != op_map.end()) {
        return it->second;
    }

    // Remove aten:: prefix for unknown ops
    if (ts_op.substr(0, 6) == "aten::") {
        return ts_op.substr(6);
    }

    return ts_op;
}

} // anonymous namespace

// ============================================================================
// TorchScriptImporter Implementation
// ============================================================================

class TorchScriptImporter::Impl {
public:
    ImportResult import_file(const std::string& path, const ImportOptions& options) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            throw ModelParseError("Cannot open file: " + path, "TorchScript");
        }

        auto size = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<uint8_t> buffer(static_cast<size_t>(size));
        if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
            throw ModelParseError("Failed to read file: " + path, "TorchScript");
        }

        return import_buffer(buffer.data(), buffer.size(), options);
    }

    ImportResult import_buffer(const void* data, size_t size, const ImportOptions& options) {
        ImportResult result;

        // Parse the TorchScript archive
        auto archive = parse_archive(data, size);

        if (!archive.is_torchscript) {
            throw ModelParseError(
                "Not a TorchScript archive. "
                "Use PyTorchImporter for regular checkpoints.",
                "TorchScript"
            );
        }

        // Create graph
        result.graph = std::make_unique<Graph>("torchscript_model");

        // Set metadata
        result.metadata.producer_name = "PyTorch TorchScript";
        result.metadata.producer_version = archive.version;

        // Parse the data pickle to get tensor metadata
        if (!archive.data.empty()) {
            try {
                pytorch::PickleParser::ParseContext data_ctx(
                    archive.data.data(), archive.data.size());
                data_ctx.storage_data = archive.tensors;  // Add tensor data

                auto parsed = pytorch::PickleParser::parse_buffer(
                    archive.data.data(), archive.data.size());

                // Load tensors as initializers
                for (const auto& [name, meta] : parsed.state_dict) {
                    try {
                        Tensor tensor = pytorch::convert_pytorch_tensor(meta);
                        result.graph->add_initializer(name, std::move(tensor));
                        result.stats.total_initializers++;
                    } catch (const std::exception& e) {
                        result.warnings.push_back(
                            "Failed to load tensor '" + name + "': " + e.what()
                        );
                    }
                }
            } catch (const std::exception& e) {
                result.warnings.push_back(
                    "Failed to parse data.pkl: " + std::string(e.what())
                );
            }
        }

        // TorchScript graph structure is typically in model.pkl
        // For a minimal implementation, we create placeholder nodes
        // A full implementation would parse the TorchScript IR

        if (!archive.model.empty()) {
            result.warnings.push_back(
                "TorchScript graph parsing not fully implemented. "
                "Only weights have been loaded."
            );
        }

        // Add a default input if specified
        for (const auto& [name, shape] : options.input_shapes) {
            TensorInfo info;
            info.name = name;
            info.shape = to_shape(shape);
            info.dtype = DType::Float32;
            result.graph->add_input(info);
        }

        return result;
    }

    bool can_import(const std::string& path) const {
        return TorchScriptImporter::is_torchscript_file(path);
    }
};

// ============================================================================
// TorchScriptImporter Public API
// ============================================================================

TorchScriptImporter::TorchScriptImporter() : impl_(std::make_unique<Impl>()) {}

TorchScriptImporter::~TorchScriptImporter() = default;

TorchScriptImporter::TorchScriptImporter(TorchScriptImporter&&) noexcept = default;
TorchScriptImporter& TorchScriptImporter::operator=(TorchScriptImporter&&) noexcept = default;

bool TorchScriptImporter::can_import(const std::string& path) const {
    return impl_->can_import(path);
}

ImportResult TorchScriptImporter::import_file(const std::string& path, const ImportOptions& options) {
    return impl_->import_file(path, options);
}

ImportResult TorchScriptImporter::import_buffer(const void* data, size_t size, const ImportOptions& options) {
    return impl_->import_buffer(data, size, options);
}

bool TorchScriptImporter::is_torchscript_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    // Read enough to check ZIP header and some contents
    std::vector<uint8_t> header(8192);
    auto bytes_read = file.read(reinterpret_cast<char*>(header.data()), header.size()).gcount();
    header.resize(static_cast<size_t>(bytes_read));

    if (header.size() < 4) {
        return false;
    }

    // Check ZIP header
    if (header[0] != 'P' || header[1] != 'K' ||
        header[2] != 0x03 || header[3] != 0x04) {
        return false;
    }

    // Look for TorchScript markers in the header portion
    // (version file, code directory, etc.)
    std::string header_str(header.begin(), header.end());
    return header_str.find("version") != std::string::npos ||
           header_str.find("bytecode.pkl") != std::string::npos ||
           header_str.find("code/") != std::string::npos;
}

// ============================================================================
// TorchScript Importer Registration
// ============================================================================

void register_torchscript_importer() {
    static bool registered = false;
    if (!registered) {
        ImporterRegistry::instance().register_importer(
            std::make_unique<TorchScriptImporter>()
        );
        registered = true;
    }
}

// Auto-register at static initialization
namespace {
    struct TorchScriptImporterRegistrar {
        TorchScriptImporterRegistrar() {
            register_torchscript_importer();
        }
    };
    static TorchScriptImporterRegistrar torchscript_importer_registrar;
}

} // namespace import
} // namespace pyflame_rt
