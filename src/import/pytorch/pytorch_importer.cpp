#include "pyflame_rt/import/pytorch_importer.hpp"
#include "pyflame_rt/import/shape_inference.hpp"
#include "pickle_parser.hpp"

namespace pyflame_rt {
namespace import {

// ============================================================================
// PyTorchImporter Implementation
// ============================================================================

class PyTorchImporter::Impl {
public:
    ModelDefiner model_definer_;
    std::string model_architecture_;

    ImportResult import_file(const std::string& path, const ImportOptions& options) {
        // Parse the checkpoint
        auto parse_result = pytorch::PickleParser::parse_file(path);
        return import_state_dict(parse_result, path, options);
    }

    ImportResult import_buffer(const void* data, size_t size, const ImportOptions& options) {
        auto parse_result = pytorch::PickleParser::parse_buffer(data, size);
        return import_state_dict(parse_result, "buffer", options);
    }

private:
    ImportResult import_state_dict(
        const pytorch::PickleParser::ParseResult& parse_result,
        const std::string& source,
        const ImportOptions& options
    ) {
        ImportResult result;

        // We need a model definer to create the graph structure
        if (!model_definer_) {
            // Without a model definer, we can only create a minimal graph
            // with initializers (weights) but no operations
            result.graph = std::make_unique<Graph>("pytorch_weights");
            result.warnings.push_back(
                "No model definer set - created graph with weights only, no operations"
            );
        } else {
            // Use the model definer to create the graph structure
            try {
                result.graph = model_definer_(source);
                if (!result.graph) {
                    throw ModelParseError("Model definer returned null graph", "PyTorch");
                }
            } catch (const std::exception& e) {
                throw ModelParseError(
                    "Model definer failed: " + std::string(e.what()),
                    "PyTorch"
                );
            }
        }

        // Set metadata
        result.metadata.producer_name = "PyTorch";
        result.metadata.graph_name = result.graph->name();

        // Copy warnings from parser
        result.warnings.insert(
            result.warnings.end(),
            parse_result.warnings.begin(),
            parse_result.warnings.end()
        );

        // Load weights from state dict
        for (const auto& [param_name, tensor_meta] : parse_result.state_dict) {
            try {
                Tensor tensor = pytorch::convert_pytorch_tensor(tensor_meta);

                // Map parameter name to initializer name
                // PyTorch uses module.submodule.weight format
                // We keep the same naming convention
                result.graph->add_initializer(param_name, std::move(tensor));

                result.stats.total_initializers++;
                result.stats.total_weight_bytes += tensor_meta.storage
                    ? tensor_meta.storage->data.size() : 0;
            } catch (const std::exception& e) {
                result.warnings.push_back(
                    "Failed to load weight '" + param_name + "': " + e.what()
                );
            }
        }

        // Run shape inference if requested and we have a graph structure
        if (options.infer_shapes && model_definer_) {
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
// PyTorchImporter Public API
// ============================================================================

PyTorchImporter::PyTorchImporter() : impl_(std::make_unique<Impl>()) {}

PyTorchImporter::~PyTorchImporter() = default;

PyTorchImporter::PyTorchImporter(PyTorchImporter&&) noexcept = default;
PyTorchImporter& PyTorchImporter::operator=(PyTorchImporter&&) noexcept = default;

ImportResult PyTorchImporter::import_file(const std::string& path, const ImportOptions& options) {
    return impl_->import_file(path, options);
}

ImportResult PyTorchImporter::import_buffer(const void* data, size_t size, const ImportOptions& options) {
    return impl_->import_buffer(data, size, options);
}

void PyTorchImporter::set_model_definer(ModelDefiner definer) {
    impl_->model_definer_ = std::move(definer);
}

bool PyTorchImporter::has_model_definer() const {
    return static_cast<bool>(impl_->model_definer_);
}

void PyTorchImporter::clear_model_definer() {
    impl_->model_definer_ = nullptr;
}

void PyTorchImporter::set_model_architecture(const std::string& arch_name) {
    impl_->model_architecture_ = arch_name;
    // Model zoo lookup not yet implemented
}

bool PyTorchImporter::is_pytorch_file(const std::string& path) {
    return pytorch::PickleParser::is_pytorch_checkpoint(path);
}

// ============================================================================
// PyTorch Importer Registration
// ============================================================================

void register_pytorch_importer() {
    static bool registered = false;
    if (!registered) {
        ImporterRegistry::instance().register_importer(
            std::make_unique<PyTorchImporter>()
        );
        registered = true;
    }
}

// Auto-register at static initialization
namespace {
    struct PyTorchImporterRegistrar {
        PyTorchImporterRegistrar() {
            register_pytorch_importer();
        }
    };
    static PyTorchImporterRegistrar pytorch_importer_registrar;
}

} // namespace import
} // namespace pyflame_rt
