#include "pyflame_rt/opt/operator_fusion.hpp"
#include "pyflame_rt/registry.hpp"

namespace pyflame_rt {
namespace opt {

OperatorFusionPass::OperatorFusionPass() {
    register_builtin_patterns();
}

OperatorFusionPass::OperatorFusionPass(FusionConfig config)
    : config_(std::move(config))
{
    register_builtin_patterns();
}

void OperatorFusionPass::register_pattern(FusionPattern pattern) {
    patterns_.push_back(std::move(pattern));
}

bool OperatorFusionPass::apply_fusion(
    Graph& graph,
    const FusionPattern& pattern,
    const PatternMatch& match)
{
    // Check constraints if provided
    if (pattern.constraint && !pattern.constraint(match, graph)) {
        return false;
    }

    // Check backend support if configured
    if (config_.check_backend_support) {
        if (!OperatorRegistry::instance().has(pattern.fused_op)) {
            return false;
        }
    }

    // Create fused node
    std::shared_ptr<Node> fused_node;
    if (pattern.fuse_func) {
        fused_node = pattern.fuse_func(match, graph);
    }

    if (!fused_node) {
        return false;
    }

    // Add fused node to graph
    graph.add_node(std::make_unique<Node>(*fused_node));

    // Remove original nodes
    for (Node* node : match.nodes) {
        if (node) {
            graph.remove_node(node->name());
        }
    }

    return true;
}

PassResult OperatorFusionPass::run(Graph& graph) {
    PassResult result;

    for (const auto& pattern : patterns_) {
        // Create matcher for this pattern
        PatternMatcher matcher(pattern.pattern);

        // Find all matches
        auto matches = matcher.match(graph);

        // Apply fusion for each match
        for (const auto& match : matches) {
            if (apply_fusion(graph, pattern, match)) {
                result.stats.nodes_fused++;
                result.modified = true;
            }
        }
    }

    return result;
}

// Separate file for built-in patterns
void OperatorFusionPass::register_builtin_patterns() {
    // Patterns are registered in fusion_patterns.cpp
}

// Register pass with global registry
REGISTER_OPT_PASS(OperatorFusionPass)

} // namespace opt
} // namespace pyflame_rt
