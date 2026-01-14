#include "pyflame_rt/opt/cse.hpp"

#include <functional>

namespace pyflame_rt {
namespace opt {

CSEPass::CSEPass(CSEConfig config)
    : config_(std::move(config)) {}

size_t CSEPass::hash_node(const Node& node) const {
    size_t hash = std::hash<std::string>{}(node.op_type());

    // Hash inputs
    for (const auto& input : node.inputs()) {
        hash ^= std::hash<std::string>{}(input) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }

    // Hash attributes if configured
    if (config_.check_attributes) {
        // Hash attribute names (values would require type-specific handling)
        for (const auto& attr_name : node.attr_names()) {
            hash ^= std::hash<std::string>{}(attr_name) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
    }

    return hash;
}

bool CSEPass::are_equivalent(const Node& a, const Node& b) const {
    // Must have same op type
    if (a.op_type() != b.op_type()) {
        return false;
    }

    // Must have same number of inputs
    if (a.inputs().size() != b.inputs().size()) {
        return false;
    }

    // Must have same inputs in same order
    for (size_t i = 0; i < a.inputs().size(); ++i) {
        if (a.inputs()[i] != b.inputs()[i]) {
            return false;
        }
    }

    // Check attributes if configured
    if (config_.check_attributes) {
        auto attrs_a = a.attr_names();
        auto attrs_b = b.attr_names();

        if (attrs_a.size() != attrs_b.size()) {
            return false;
        }

        // Sort attribute names for comparison
        std::sort(attrs_a.begin(), attrs_a.end());
        std::sort(attrs_b.begin(), attrs_b.end());

        if (attrs_a != attrs_b) {
            return false;
        }

        // Compare attribute values
        // Note: This is a simplified check - full implementation would need
        // to compare actual values which requires handling different types
        for (const auto& attr_name : attrs_a) {
            // For now, we assume if names match and types match, values match
            // A full implementation would compare actual values
        }
    }

    return true;
}

PassResult CSEPass::run(Graph& graph) {
    PassResult result;

    // Build hash table for quick lookups
    std::unordered_map<size_t, std::vector<Node*>> hash_buckets;

    for (const auto& node : graph.nodes()) {
        size_t h = hash_node(*node);
        hash_buckets[h].push_back(node.get());
    }

    // Find equivalent nodes
    std::unordered_map<std::string, std::string> rewrites;  // old output -> new output
    std::vector<std::string> nodes_to_remove;
    size_t comparisons = 0;

    for (auto& [hash, nodes] : hash_buckets) {
        if (nodes.size() < 2) continue;

        // Compare nodes in this bucket
        for (size_t i = 0; i < nodes.size() && comparisons < config_.max_comparisons; ++i) {
            Node* node_i = nodes[i];

            // Skip if already marked for removal
            if (std::find(nodes_to_remove.begin(), nodes_to_remove.end(),
                          node_i->name()) != nodes_to_remove.end()) {
                continue;
            }

            for (size_t j = i + 1; j < nodes.size() && comparisons < config_.max_comparisons; ++j) {
                Node* node_j = nodes[j];
                comparisons++;

                // Skip if already marked for removal
                if (std::find(nodes_to_remove.begin(), nodes_to_remove.end(),
                              node_j->name()) != nodes_to_remove.end()) {
                    continue;
                }

                if (are_equivalent(*node_i, *node_j)) {
                    // node_j is redundant - use node_i's outputs instead
                    for (size_t k = 0; k < node_j->outputs().size() &&
                                       k < node_i->outputs().size(); ++k) {
                        rewrites[node_j->outputs()[k]] = node_i->outputs()[k];
                    }
                    nodes_to_remove.push_back(node_j->name());
                }
            }
        }
    }

    // Apply rewrites
    for (const auto& node : graph.nodes()) {
        // Skip nodes being removed
        if (std::find(nodes_to_remove.begin(), nodes_to_remove.end(),
                      node->name()) != nodes_to_remove.end()) {
            continue;
        }

        bool modified = false;
        std::vector<std::string> new_inputs;
        new_inputs.reserve(node->inputs().size());

        for (const auto& input : node->inputs()) {
            auto it = rewrites.find(input);
            if (it != rewrites.end()) {
                new_inputs.push_back(it->second);
                modified = true;
            } else {
                new_inputs.push_back(input);
            }
        }

        if (modified) {
            const_cast<Node*>(node.get())->set_inputs(std::move(new_inputs));
            result.modified = true;
        }
    }

    // Remove redundant nodes
    for (const auto& name : nodes_to_remove) {
        graph.remove_node(name);
        result.stats.nodes_removed++;
        result.modified = true;
    }

    return result;
}

// Register pass with global registry
REGISTER_OPT_PASS(CSEPass)

} // namespace opt
} // namespace pyflame_rt
