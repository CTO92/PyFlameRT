#include "pyflame_rt/opt/pass.hpp"
#include "pyflame_rt/errors.hpp"

#include <algorithm>
#include <iostream>
#include <queue>
#include <unordered_set>

namespace pyflame_rt {
namespace opt {

// ============================================================================
// PassRegistry
// ============================================================================

PassRegistry& PassRegistry::instance() {
    static PassRegistry registry;
    return registry;
}

void PassRegistry::register_pass_factory(
    const std::string& name,
    std::function<std::unique_ptr<Pass>()> factory)
{
    factories_[name] = std::move(factory);
}

std::unique_ptr<Pass> PassRegistry::create(const std::string& name) const {
    auto it = factories_.find(name);
    if (it != factories_.end()) {
        return it->second();
    }
    return nullptr;
}

std::vector<std::string> PassRegistry::list() const {
    std::vector<std::string> names;
    names.reserve(factories_.size());
    for (const auto& [name, _] : factories_) {
        names.push_back(name);
    }
    return names;
}

bool PassRegistry::has(const std::string& name) const {
    return factories_.find(name) != factories_.end();
}

// ============================================================================
// PassManager
// ============================================================================

PassManager::PassManager(PassManagerConfig config)
    : config_(std::move(config)) {}

PassManager::~PassManager() = default;

PassManager::PassManager(PassManager&&) noexcept = default;
PassManager& PassManager::operator=(PassManager&&) noexcept = default;

void PassManager::register_pass(std::unique_ptr<Pass> pass) {
    const char* name = pass->name();
    pass_map_[name] = pass.get();
    passes_.push_back(std::move(pass));
}

std::vector<std::string> PassManager::list_passes() const {
    std::vector<std::string> names;
    names.reserve(passes_.size());
    for (const auto& pass : passes_) {
        names.push_back(pass->name());
    }
    return names;
}

bool PassManager::has_pass(const std::string& name) const {
    return pass_map_.find(name) != pass_map_.end();
}

Pass* PassManager::get_pass(const std::string& name) {
    auto it = pass_map_.find(name);
    return (it != pass_map_.end()) ? it->second : nullptr;
}

bool PassManager::should_skip(const Pass& pass) const {
    // Check skip list
    for (const auto& skip : config_.skip_passes) {
        if (skip == pass.name()) return true;
    }

    // Check only list
    if (!config_.only_passes.empty()) {
        bool found = false;
        for (const auto& only : config_.only_passes) {
            if (only == pass.name()) {
                found = true;
                break;
            }
        }
        if (!found) return true;
    }

    // Check optimization level
    if (!pass.should_run(config_.opt_level)) return true;

    return false;
}

std::vector<Pass*> PassManager::topological_sort_passes() const {
    // Build dependency graph
    std::unordered_map<std::string, std::vector<std::string>> reverse_deps;
    std::unordered_map<std::string, int> in_degree;

    // Initialize
    for (const auto& pass : passes_) {
        std::string name = pass->name();
        in_degree[name] = 0;
    }

    // Calculate in-degrees based on dependencies
    for (const auto& pass : passes_) {
        std::string name = pass->name();
        for (const auto& dep : pass->dependencies()) {
            if (pass_map_.count(dep)) {
                reverse_deps[dep].push_back(name);
                in_degree[name]++;
            }
        }
    }

    // Kahn's algorithm
    std::queue<std::string> ready;
    for (const auto& [name, degree] : in_degree) {
        if (degree == 0) ready.push(name);
    }

    std::vector<Pass*> result;
    while (!ready.empty()) {
        std::string name = ready.front();
        ready.pop();

        auto it = pass_map_.find(name);
        if (it != pass_map_.end()) {
            result.push_back(it->second);
        }

        // Reduce in-degree of dependents
        for (const auto& dependent : reverse_deps[name]) {
            if (--in_degree[dependent] == 0) {
                ready.push(dependent);
            }
        }
    }

    // Check for cycles (if result size doesn't match)
    if (result.size() != passes_.size()) {
        // Some passes couldn't be sorted - missing dependencies or cycles
        // Just return in registration order
        result.clear();
        for (const auto& pass : passes_) {
            result.push_back(pass.get());
        }
    }

    return result;
}

PassResult PassManager::run_pass(const std::string& name, Graph& graph) {
    auto it = pass_map_.find(name);
    if (it == pass_map_.end()) {
        throw PyFlameRTError("Unknown optimization pass: " + name);
    }

    Pass* pass = it->second;

    if (config_.verbose) {
        std::cout << "[opt] Running pass: " << name << std::endl;
    }

    PassResult result = pass->run(graph);

    if (config_.validate_after_pass && result.modified) {
        auto errors = graph.validate();
        if (!errors.empty()) {
            throw ValidationError(errors);
        }
    }

    if (config_.verbose && result.modified) {
        std::cout << "[opt]   Modified: "
                  << result.stats.nodes_removed << " removed, "
                  << result.stats.nodes_added << " added, "
                  << result.stats.nodes_fused << " fused, "
                  << result.stats.constants_folded << " constants folded"
                  << std::endl;
    }

    return result;
}

PassResult PassManager::run(Graph& graph) {
    PassResult total_result;

    auto sorted_passes = topological_sort_passes();

    for (Pass* pass : sorted_passes) {
        if (should_skip(*pass)) {
            if (config_.verbose) {
                std::cout << "[opt] Skipping pass: " << pass->name() << std::endl;
            }
            continue;
        }

        if (config_.verbose) {
            std::cout << "[opt] Running pass: " << pass->name() << std::endl;
        }

        PassResult result = pass->run(graph);
        total_result += result;

        if (config_.validate_after_pass && result.modified) {
            auto errors = graph.validate();
            if (!errors.empty()) {
                throw ValidationError(errors);
            }
        }

        if (config_.verbose && result.modified) {
            std::cout << "[opt]   Modified: "
                      << result.stats.nodes_removed << " removed, "
                      << result.stats.nodes_added << " added"
                      << std::endl;
        }
    }

    return total_result;
}

PassResult PassManager::run_until_fixed_point(Graph& graph) {
    PassResult total_result;

    for (int iter = 0; iter < config_.max_iterations; ++iter) {
        if (config_.verbose) {
            std::cout << "[opt] Fixed-point iteration " << (iter + 1) << std::endl;
        }

        PassResult iter_result = run(graph);
        total_result += iter_result;

        if (!iter_result.modified) {
            if (config_.verbose) {
                std::cout << "[opt] Fixed point reached after "
                          << (iter + 1) << " iterations" << std::endl;
            }
            break;
        }
    }

    return total_result;
}

PassManager PassManager::create_default(OptLevel level) {
    PassManager pm(PassManagerConfig{.opt_level = level});

    // Create passes from registry
    auto& registry = PassRegistry::instance();
    for (const auto& name : registry.list()) {
        auto pass = registry.create(name);
        if (pass) {
            pm.register_pass(std::move(pass));
        }
    }

    return pm;
}

} // namespace opt
} // namespace pyflame_rt
