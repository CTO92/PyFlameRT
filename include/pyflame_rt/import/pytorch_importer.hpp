#pragma once

#include "pyflame_rt/import/importer.hpp"

#include <functional>
#include <memory>

namespace pyflame_rt {
namespace import {

/// Callback type for defining model architecture
/// Users provide this to create the graph structure from a PyTorch checkpoint.
/// The callback receives the checkpoint path and returns a Graph with the
/// model architecture (nodes without weights). The importer then loads
/// weights from the checkpoint into the graph's initializers.
using ModelDefiner = std::function<std::unique_ptr<Graph>(
    const std::string& checkpoint_path
)>;

/// PyTorch checkpoint importer
///
/// Imports PyTorch .pt/.pth files containing state dictionaries.
/// Unlike TorchScript, regular PyTorch checkpoints only contain weights,
/// not the model architecture. Therefore, this importer requires either:
///
/// 1. A ModelDefiner callback that creates the graph structure
/// 2. A pre-defined architecture name from the model zoo (future)
///
/// Features:
/// - State dictionary parsing (weights and biases)
/// - Tensor format conversion (PyTorch storage to PyFlameRT Tensor)
/// - dtype conversion (PyTorch types to PyFlameRT DType)
/// - Support for nested state dicts (module hierarchies)
///
/// Example:
///     PyTorchImporter importer;
///
///     // Define model architecture
///     importer.set_model_definer([](const std::string& path) {
///         auto graph = std::make_unique<Graph>("my_model");
///         // Add nodes for your model architecture
///         // ...
///         return graph;
///     });
///
///     auto result = importer.import_file("model.pt");
///
class PyTorchImporter : public ModelImporter {
public:
    PyTorchImporter();
    ~PyTorchImporter() override;

    // Non-copyable
    PyTorchImporter(const PyTorchImporter&) = delete;
    PyTorchImporter& operator=(const PyTorchImporter&) = delete;

    // Movable
    PyTorchImporter(PyTorchImporter&&) noexcept;
    PyTorchImporter& operator=(PyTorchImporter&&) noexcept;

    /// Get importer name
    const char* name() const override { return "PyTorch"; }

    /// Get supported file extensions
    std::vector<std::string> extensions() const override {
        return {".pt", ".pth", ".bin"};
    }

    /// Import PyTorch checkpoint from file
    /// @note Requires model_definer to be set
    ImportResult import_file(
        const std::string& path,
        const ImportOptions& options = {}
    ) override;

    /// Import PyTorch checkpoint from memory buffer
    /// @note Requires model_definer to be set
    ImportResult import_buffer(
        const void* data,
        size_t size,
        const ImportOptions& options = {}
    ) override;

    /// Set the model definition callback
    /// This callback creates the graph architecture that weights will be loaded into
    void set_model_definer(ModelDefiner definer);

    /// Check if a model definer has been set
    bool has_model_definer() const;

    /// Clear the model definer
    void clear_model_definer();

    /// Set model architecture by name (from model zoo)
    /// @param arch_name Architecture name (e.g., "resnet18", "bert-base")
    /// @note Not yet implemented - will be added when model zoo is available
    void set_model_architecture(const std::string& arch_name);

    /// Check if file is a valid PyTorch checkpoint
    /// @param path Path to file
    /// @return true if file appears to be a PyTorch checkpoint
    static bool is_pytorch_file(const std::string& path);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/// Register PyTorch importer with the importer registry
/// Called automatically at static initialization
void register_pytorch_importer();

} // namespace import
} // namespace pyflame_rt
