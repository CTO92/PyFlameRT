#pragma once

#include "pyflame_rt/import/importer.hpp"

#include <memory>

namespace pyflame_rt {
namespace import {

/// TorchScript model importer
///
/// Imports TorchScript models (.pt, .pts files containing JIT traces/scripts).
/// Unlike regular PyTorch checkpoints, TorchScript models are self-contained
/// and include both the model architecture (as TorchScript IR) and weights.
///
/// Features:
/// - Support for both traced (torch.jit.trace) and scripted (torch.jit.script) models
/// - Automatic graph structure extraction from TorchScript IR
/// - Operator conversion from TorchScript ops to PyFlameRT ops
/// - Control flow operators (loops, conditionals) via subgraph support
///
/// Limitations:
/// - Some TorchScript-specific operators may not be supported
/// - Dynamic control flow may be limited
///
/// Example:
///     TorchScriptImporter importer;
///     auto result = importer.import_file("model.pt");
///     if (result.success()) {
///         // Use the imported graph directly
///     }
///
class TorchScriptImporter : public ModelImporter {
public:
    TorchScriptImporter();
    ~TorchScriptImporter() override;

    // Non-copyable
    TorchScriptImporter(const TorchScriptImporter&) = delete;
    TorchScriptImporter& operator=(const TorchScriptImporter&) = delete;

    // Movable
    TorchScriptImporter(TorchScriptImporter&&) noexcept;
    TorchScriptImporter& operator=(TorchScriptImporter&&) noexcept;

    /// Get importer name
    const char* name() const override { return "TorchScript"; }

    /// Get supported file extensions
    /// Note: .pt files can be either TorchScript or regular checkpoints
    /// The importer checks the file format to determine which it is
    std::vector<std::string> extensions() const override {
        return {".pts", ".pt"};
    }

    /// Check if file can be imported as TorchScript
    /// @note Returns false for regular PyTorch checkpoints
    bool can_import(const std::string& path) const override;

    /// Import TorchScript model from file
    ImportResult import_file(
        const std::string& path,
        const ImportOptions& options = {}
    ) override;

    /// Import TorchScript model from memory buffer
    ImportResult import_buffer(
        const void* data,
        size_t size,
        const ImportOptions& options = {}
    ) override;

    /// Check if file is a TorchScript model (vs regular checkpoint)
    static bool is_torchscript_file(const std::string& path);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/// Register TorchScript importer with the importer registry
/// Called automatically at static initialization
void register_torchscript_importer();

} // namespace import
} // namespace pyflame_rt
