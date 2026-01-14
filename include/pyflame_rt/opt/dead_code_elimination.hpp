#pragma once

#include "pyflame_rt/opt/pass.hpp"

#include <unordered_set>

namespace pyflame_rt {
namespace opt {

/// Dead code elimination configuration
struct DCEConfig {
    /// Remove unused initializers
    bool remove_initializers = true;

    /// Remove identity nodes (where output == input)
    bool remove_identity = true;

    /// Remove dropout nodes in inference mode
    bool remove_dropout = true;
};

/// Dead code elimination pass
///
/// Removes nodes whose outputs are not used by any other node
/// or by the graph outputs. Also removes unused initializers.
///
/// Algorithm:
/// 1. Mark all graph outputs as "live"
/// 2. Traverse backwards, marking inputs of live nodes as live
/// 3. Remove all non-live nodes and initializers
///
class DeadCodeEliminationPass : public Pass {
public:
    DeadCodeEliminationPass() = default;
    explicit DeadCodeEliminationPass(DCEConfig config);

    const char* name() const override { return "DeadCodeElimination"; }

    const char* description() const override {
        return "Remove unused nodes and initializers";
    }

    PassResult run(Graph& graph) override;

    bool should_run(OptLevel level) const override {
        return level >= OptLevel::Basic;
    }

    void set_config(const DCEConfig& config) { config_ = config; }
    const DCEConfig& config() const { return config_; }

private:
    DCEConfig config_;

    /// Find all tensors needed to compute outputs
    std::unordered_set<std::string> find_live_tensors(const Graph& graph) const;

    /// Check if node should be removed (identity, dropout)
    bool should_remove_node(const Graph& graph, const Node& node) const;
};

} // namespace opt
} // namespace pyflame_rt
