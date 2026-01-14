#pragma once

#include "pyflame_rt/graph.hpp"
#include <memory>
#include <string>
#include <cstdint>

namespace pyflame_rt {

/// Security limits for model deserialization
struct DeserializationLimits {
    static constexpr uint64_t MAX_STRING_SIZE = 10ULL * 1024 * 1024;        // 10 MB
    static constexpr uint64_t MAX_TENSOR_DATA_SIZE = 4ULL * 1024 * 1024 * 1024;  // 4 GB
    static constexpr uint64_t MAX_ARRAY_COUNT = 1000000;                     // 1M elements
    static constexpr uint64_t MAX_DIMS = 16;                                 // Max tensor dimensions
    static constexpr uint64_t MAX_NODES = 1000000;                           // Max graph nodes
    static constexpr uint64_t MAX_ATTRIBUTES = 1000;                         // Max attributes per node
    static constexpr int64_t MAX_DIMENSION_SIZE = 1000000000LL;              // 1 billion per dimension (CRIT-02 fix)
};

/// Native PyFlame model format (.pfm) handler
class PyFlameFormat {
public:
    static constexpr char MAGIC[4] = {'P', 'F', 'M', '\0'};
    static constexpr uint32_t VERSION = 1;

    /// Load a .pfm file and return Graph
    static std::unique_ptr<Graph> load(const std::string& path);

    /// Save Graph to .pfm file
    static void save(const Graph& graph, const std::string& path);
};

} // namespace pyflame_rt
