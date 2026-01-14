#pragma once

#include "pyflame_rt/opt/pass.hpp"

#include <unordered_map>

namespace pyflame_rt {
namespace opt {

/// CSE configuration
struct CSEConfig {
    /// Include attributes in equivalence check
    bool check_attributes = true;

    /// Maximum comparisons per node (performance limit)
    size_t max_comparisons = 10000;
};

/// Common subexpression elimination pass
///
/// Identifies nodes that compute the same value and eliminates
/// redundant computation by reusing results.
///
/// Two nodes are considered equivalent if:
/// 1. They have the same op_type
/// 2. They have the same inputs (in the same order)
/// 3. They have the same attributes (if check_attributes is true)
///
class CSEPass : public Pass {
public:
    CSEPass() = default;
    explicit CSEPass(CSEConfig config);

    const char* name() const override { return "CSE"; }

    const char* description() const override {
        return "Eliminate redundant computations";
    }

    PassResult run(Graph& graph) override;

    bool should_run(OptLevel level) const override {
        return level >= OptLevel::Extended;
    }

    void set_config(const CSEConfig& config) { config_ = config; }
    const CSEConfig& config() const { return config_; }

private:
    CSEConfig config_;

    /// Compute hash for node (for fast comparison)
    size_t hash_node(const Node& node) const;

    /// Check if two nodes are equivalent
    bool are_equivalent(const Node& a, const Node& b) const;
};

} // namespace opt
} // namespace pyflame_rt
