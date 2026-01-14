#pragma once

#include "pyflame_rt/graph.hpp"
#include "pyflame_rt/types.hpp"
#include "pyflame_rt/errors.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <functional>

namespace pyflame_rt {
namespace import {

/// Options for model import
struct ImportOptions {
    /// Input shapes for dynamic dimensions (name -> concrete shape)
    std::unordered_map<std::string, std::vector<int64_t>> input_shapes;

    /// Whether to run shape inference after import
    bool infer_shapes = true;

    /// Whether to load initializers (weights) immediately
    bool load_initializers = true;

    /// Opset version override (0 = use model's opset)
    int opset_version = 0;

    /// Whether to allow unsupported operators (creates placeholder nodes)
    bool allow_unsupported = false;

    /// Custom operator domain mappings
    std::unordered_map<std::string, std::string> op_domain_map;
};

/// Statistics from import operation
struct ImportStats {
    int total_ops = 0;
    int mapped_ops = 0;
    int unsupported_ops = 0;
    std::vector<std::string> unsupported_op_types;
    int total_initializers = 0;
    size_t total_weight_bytes = 0;
};

/// Result of model import
struct ImportResult {
    /// The imported graph (nullptr on failure)
    std::unique_ptr<Graph> graph;

    /// Model metadata
    ModelMetadata metadata;

    /// Import statistics
    ImportStats stats;

    /// Warnings encountered during import
    std::vector<std::string> warnings;

    /// Check if import succeeded
    bool success() const { return graph != nullptr; }
};

/// Abstract base class for model importers
///
/// All format-specific importers inherit from this class and implement
/// the import methods for their particular format.
class ModelImporter {
public:
    virtual ~ModelImporter() = default;

    /// Get importer name (e.g., "ONNX", "PyTorch", "TorchScript")
    virtual const char* name() const = 0;

    /// Get supported file extensions (e.g., {".onnx"}, {".pt", ".pth"})
    virtual std::vector<std::string> extensions() const = 0;

    /// Check if file can be imported by this importer
    /// Default implementation checks file extension
    virtual bool can_import(const std::string& path) const;

    /// Import model from file
    /// @param path Path to model file
    /// @param options Import options
    /// @return Import result with graph and metadata
    virtual ImportResult import_file(
        const std::string& path,
        const ImportOptions& options = {}
    ) = 0;

    /// Import model from memory buffer
    /// @param data Pointer to model data
    /// @param size Size of data in bytes
    /// @param options Import options
    /// @return Import result with graph and metadata
    virtual ImportResult import_buffer(
        const void* data,
        size_t size,
        const ImportOptions& options = {}
    ) = 0;

protected:
    /// Helper to check file extension
    bool has_extension(const std::string& path, const std::string& ext) const;

    /// Helper to get file extension
    std::string get_extension(const std::string& path) const;
};

/// Registry of available model importers
///
/// Singleton that manages all registered importers and dispatches
/// import requests to the appropriate importer based on file extension.
class ImporterRegistry {
public:
    /// Get the singleton instance
    static ImporterRegistry& instance();

    /// Register an importer
    /// @param importer Unique pointer to importer (registry takes ownership)
    void register_importer(std::unique_ptr<ModelImporter> importer);

    /// Get importer for file path (based on extension)
    /// @return Pointer to importer, or nullptr if none found
    ModelImporter* get(const std::string& path) const;

    /// Get importer by name
    /// @return Pointer to importer, or nullptr if none found
    ModelImporter* get_by_name(const std::string& name) const;

    /// List all registered importer names
    std::vector<std::string> list() const;

    /// List all supported file extensions
    std::vector<std::string> supported_extensions() const;

    /// Check if any importer supports the file
    bool can_import(const std::string& path) const;

private:
    ImporterRegistry() = default;
    ImporterRegistry(const ImporterRegistry&) = delete;
    ImporterRegistry& operator=(const ImporterRegistry&) = delete;

    std::vector<std::unique_ptr<ModelImporter>> importers_;
};

/// Convenience function to import a model using auto-detected format
/// @param path Path to model file
/// @param options Import options
/// @return Import result
/// @throws UnsupportedFormatError if format not recognized
ImportResult import_model(
    const std::string& path,
    const ImportOptions& options = {}
);

// ============================================================================
// Import-specific exceptions
// ============================================================================

/// Raised when model parsing fails
class ModelParseError : public InvalidModelError {
public:
    ModelParseError(const std::string& message,
                    const std::string& format,
                    int line = -1)
        : InvalidModelError(build_message(message, format, line))
        , format_(format)
        , line_(line) {}

    const std::string& format() const { return format_; }
    int line() const { return line_; }

private:
    static std::string build_message(const std::string& message,
                                     const std::string& format,
                                     int line) {
        std::string msg = "[" + format + "] " + message;
        if (line >= 0) {
            msg += " (line " + std::to_string(line) + ")";
        }
        return msg;
    }

    std::string format_;
    int line_;
};

/// Raised when an operator cannot be converted
class OperatorConversionError : public PyFlameRTError {
public:
    OperatorConversionError(const std::string& op_name,
                            const std::string& framework,
                            const std::string& reason)
        : PyFlameRTError("Cannot convert " + framework + " operator '" +
                         op_name + "': " + reason)
        , op_name_(op_name)
        , framework_(framework) {}

    const std::string& op_name() const { return op_name_; }
    const std::string& framework() const { return framework_; }

private:
    std::string op_name_;
    std::string framework_;
};

} // namespace import
} // namespace pyflame_rt
