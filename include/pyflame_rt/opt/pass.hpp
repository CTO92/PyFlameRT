#pragma once

#include "pyflame_rt/graph.hpp"
#include "pyflame_rt/types.hpp"

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace pyflame_rt {
namespace opt {

/// Statistics from an optimization pass
struct PassStats {
    int nodes_removed = 0;
    int nodes_added = 0;
    int nodes_fused = 0;
    int constants_folded = 0;
    int initializers_removed = 0;

    PassStats& operator+=(const PassStats& other) {
        nodes_removed += other.nodes_removed;
        nodes_added += other.nodes_added;
        nodes_fused += other.nodes_fused;
        constants_folded += other.constants_folded;
        initializers_removed += other.initializers_removed;
        return *this;
    }
};

/// Result of an optimization pass
struct PassResult {
    /// Whether the pass modified the graph
    bool modified = false;

    /// Whether optimization was aborted due to timeout (MED-01 security fix)
    bool timed_out = false;

    /// Statistics about changes
    PassStats stats;

    /// Warnings generated during the pass
    std::vector<std::string> warnings;

    PassResult& operator+=(const PassResult& other) {
        modified = modified || other.modified;
        timed_out = timed_out || other.timed_out;
        stats += other.stats;
        warnings.insert(warnings.end(),
                       other.warnings.begin(),
                       other.warnings.end());
        return *this;
    }
};

/// Abstract base class for optimization passes
class Pass {
public:
    virtual ~Pass() = default;

    /// Get pass name (must be unique)
    virtual const char* name() const = 0;

    /// Get human-readable description
    virtual const char* description() const = 0;

    /// Run the pass on the graph
    /// @param graph The graph to optimize (modified in place)
    /// @return Result indicating changes made
    virtual PassResult run(Graph& graph) = 0;

    /// Check if pass should run at given optimization level
    virtual bool should_run(OptLevel level) const { return true; }

    /// Get dependencies (names of passes that must run first)
    virtual std::vector<std::string> dependencies() const { return {}; }

    /// Whether this pass can be run multiple times
    virtual bool is_repeatable() const { return true; }
};

/// Pass manager configuration
struct PassManagerConfig {
    /// Optimization level
    OptLevel opt_level = OptLevel::Extended;

    /// Maximum iterations for fixed-point optimization
    int max_iterations = 10;

    /// Timeout in milliseconds (0 = no timeout) - MED-01 security fix
    int timeout_ms = 30000;  // Default 30 seconds

    /// Enable verbose logging
    bool verbose = false;

    /// Passes to skip (by name)
    std::vector<std::string> skip_passes;

    /// Only run these passes (empty = all applicable)
    std::vector<std::string> only_passes;

    /// Validate graph after each pass
    bool validate_after_pass = true;
};

/// Manages and executes optimization passes
class PassManager {
public:
    explicit PassManager(PassManagerConfig config = {});
    ~PassManager();

    // Non-copyable
    PassManager(const PassManager&) = delete;
    PassManager& operator=(const PassManager&) = delete;

    // Movable
    PassManager(PassManager&&) noexcept;
    PassManager& operator=(PassManager&&) noexcept;

    /// Register a pass
    void register_pass(std::unique_ptr<Pass> pass);

    /// Run all applicable passes on the graph
    PassResult run(Graph& graph);

    /// Run a specific pass by name
    /// @throws std::runtime_error if pass not found
    PassResult run_pass(const std::string& name, Graph& graph);

    /// Run passes until no changes (fixed-point)
    PassResult run_until_fixed_point(Graph& graph);

    /// Get list of registered pass names
    std::vector<std::string> list_passes() const;

    /// Check if pass is registered
    bool has_pass(const std::string& name) const;

    /// Get pass by name
    Pass* get_pass(const std::string& name);

    /// Create default pass manager with built-in passes
    static PassManager create_default(OptLevel level = OptLevel::Extended);

    /// Get/set configuration
    const PassManagerConfig& config() const { return config_; }
    void set_config(const PassManagerConfig& config) { config_ = config; }

private:
    PassManagerConfig config_;
    std::vector<std::unique_ptr<Pass>> passes_;
    std::unordered_map<std::string, Pass*> pass_map_;

    /// Sort passes by dependencies
    std::vector<Pass*> topological_sort_passes() const;

    /// Check if pass should be skipped
    bool should_skip(const Pass& pass) const;
};

/// Global pass registry (for automatic registration)
class PassRegistry {
public:
    static PassRegistry& instance();

    void register_pass_factory(const std::string& name,
                               std::function<std::unique_ptr<Pass>()> factory);

    std::unique_ptr<Pass> create(const std::string& name) const;
    std::vector<std::string> list() const;
    bool has(const std::string& name) const;

private:
    PassRegistry() = default;
    std::unordered_map<std::string, std::function<std::unique_ptr<Pass>()>> factories_;
};

/// Helper macro for static pass registration
#define REGISTER_OPT_PASS(PassClass) \
    namespace { \
        struct PassClass##Registrar { \
            PassClass##Registrar() { \
                ::pyflame_rt::opt::PassRegistry::instance() \
                    .register_pass_factory(#PassClass, \
                        []() { return std::make_unique<PassClass>(); }); \
            } \
        }; \
        static PassClass##Registrar _pass_registrar_##PassClass; \
    }

} // namespace opt
} // namespace pyflame_rt
