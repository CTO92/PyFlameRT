#pragma once

#include "pyflame_rt/graph.hpp"
#include <memory>
#include <string>
#include <cstddef>

namespace pyflame_rt {
namespace io {

/// Load a model graph from file
std::unique_ptr<Graph> load_model(const std::string& path);

/// Load a model graph from memory buffer
std::unique_ptr<Graph> load_model_from_buffer(const void* data, size_t size);

/// Save a model graph to file
bool save_model(const std::string& path, const Graph& graph);

/// Save a model graph to memory buffer
std::vector<uint8_t> save_model_to_buffer(const Graph& graph);

/// Get file extension (lowercase)
std::string get_extension(const std::string& path);

/// Check if a file format is supported
bool is_supported_format(const std::string& path);

} // namespace io
} // namespace pyflame_rt
