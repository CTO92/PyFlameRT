#pragma once

#include "pyflame_rt/import/importer.hpp"

namespace pyflame_rt {
namespace import {

/// ONNX model importer
///
/// Imports ONNX models (.onnx files) into PyFlameRT graph format.
/// Supports ONNX opsets 9-21 with automatic operator version handling.
///
/// Features:
/// - Full graph structure parsing (nodes, inputs, outputs)
/// - Initializer (weights) loading
/// - Operator mapping to PyFlameRT operators
/// - Shape inference integration
/// - Subgraph support (for control flow operators)
///
/// Example:
///     ONNXImporter importer;
///     auto result = importer.import_file("model.onnx");
///     if (result.success()) {
///         auto& graph = result.graph;
///         // Use the imported graph
///     }
///
class ONNXImporter : public ModelImporter {
public:
    ONNXImporter();
    ~ONNXImporter() override;

    // Non-copyable
    ONNXImporter(const ONNXImporter&) = delete;
    ONNXImporter& operator=(const ONNXImporter&) = delete;

    // Movable
    ONNXImporter(ONNXImporter&&) noexcept;
    ONNXImporter& operator=(ONNXImporter&&) noexcept;

    /// Get importer name
    const char* name() const override { return "ONNX"; }

    /// Get supported file extensions
    std::vector<std::string> extensions() const override {
        return {".onnx"};
    }

    /// Import ONNX model from file
    ImportResult import_file(
        const std::string& path,
        const ImportOptions& options = {}
    ) override;

    /// Import ONNX model from memory buffer
    ImportResult import_buffer(
        const void* data,
        size_t size,
        const ImportOptions& options = {}
    ) override;

    /// Get ONNX opset version from a model file
    /// @param path Path to ONNX model
    /// @return Opset version, or -1 on error
    static int get_opset_version(const std::string& path);

    /// Minimum supported ONNX opset version
    static constexpr int MIN_OPSET_VERSION = 9;

    /// Maximum supported ONNX opset version
    static constexpr int MAX_OPSET_VERSION = 21;

    /// Check if opset version is supported
    static bool is_opset_supported(int version) {
        return version >= MIN_OPSET_VERSION && version <= MAX_OPSET_VERSION;
    }

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/// Register ONNX importer with the importer registry
/// Called automatically at static initialization
void register_onnx_importer();

} // namespace import
} // namespace pyflame_rt
