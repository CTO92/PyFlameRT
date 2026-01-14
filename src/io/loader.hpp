#pragma once

#include "pyflame_rt/graph.hpp"
#include <memory>
#include <string>

namespace pyflame_rt {

/// Load a model graph from file
std::unique_ptr<Graph> load_model(const std::string& path);

/// Save a model graph to file
void save_model(const Graph& graph, const std::string& path);

/// Get file extension
std::string get_extension(const std::string& path);

} // namespace pyflame_rt
