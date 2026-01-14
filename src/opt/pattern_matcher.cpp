#include "pyflame_rt/opt/pattern_matcher.hpp"

#include <stdexcept>

namespace pyflame_rt {
namespace opt {

// ============================================================================
// PatternNode
// ============================================================================

PatternNode PatternNode::op(const std::string& op_type, const std::string& capture) {
    PatternNode node;
    node.op_type = op_type;
    node.capture = capture;
    node.optional = false;
    return node;
}

PatternNode PatternNode::any(const std::string& capture) {
    PatternNode node;
    node.op_type = "";  // Wildcard
    node.capture = capture;
    node.optional = false;
    return node;
}

PatternNode PatternNode::matching(NodePredicate pred, const std::string& capture) {
    PatternNode node;
    node.op_type = "";
    node.predicate = std::move(pred);
    node.capture = capture;
    node.optional = false;
    return node;
}

// ============================================================================
// Pattern
// ============================================================================

int Pattern::add_node(PatternNode node) {
    int idx = static_cast<int>(nodes_.size());
    nodes_.push_back(std::move(node));
    return idx;
}

void Pattern::add_edge(int from, int from_output, int to, int to_input) {
    edges_.push_back(PatternEdge{from, from_output, to, to_input});
}

void Pattern::add_edge(int from, int to) {
    add_edge(from, 0, to, 0);
}

void Pattern::set_root(int node_idx) {
    root_ = node_idx;
}

Pattern Pattern::sequence(const std::vector<std::string>& op_types) {
    Pattern p;

    if (op_types.empty()) {
        return p;
    }

    int prev = -1;
    for (size_t i = 0; i < op_types.size(); ++i) {
        std::string capture = "node" + std::to_string(i);
        int curr = p.add_node(PatternNode::op(op_types[i], capture));

        if (prev >= 0) {
            // Previous node's output feeds into current node's input
            p.add_edge(prev, curr);
        }
        prev = curr;
    }

    // Root is the last node in sequence (output)
    p.set_root(static_cast<int>(op_types.size()) - 1);

    return p;
}

Pattern Pattern::single(const std::string& op_type) {
    Pattern p;
    int idx = p.add_node(PatternNode::op(op_type, "node0"));
    p.set_root(idx);
    return p;
}

Pattern Pattern::single(const std::string& op_type, NodePredicate pred) {
    Pattern p;
    PatternNode node;
    node.op_type = op_type;
    node.predicate = std::move(pred);
    node.capture = "node0";
    int idx = p.add_node(std::move(node));
    p.set_root(idx);
    return p;
}

// ============================================================================
// PatternMatch
// ============================================================================

Node* PatternMatch::get(const std::string& name) const {
    auto it = captures.find(name);
    if (it == captures.end()) {
        throw std::runtime_error("Pattern capture not found: " + name);
    }
    return it->second;
}

bool PatternMatch::has(const std::string& name) const {
    return captures.find(name) != captures.end();
}

// ============================================================================
// PatternMatcher
// ============================================================================

PatternMatcher::PatternMatcher(const Pattern& pattern)
    : pattern_(pattern) {}

bool PatternMatcher::node_matches(const Node& node, const PatternNode& pnode) const {
    // Check operator type (empty = wildcard)
    if (!pnode.op_type.empty() && node.op_type() != pnode.op_type) {
        return false;
    }

    // Check custom predicate
    if (pnode.predicate && !pnode.predicate(node)) {
        return false;
    }

    return true;
}

bool PatternMatcher::match_node(
    const Graph& graph,
    Node* graph_node,
    int pattern_idx,
    PatternMatch& result,
    std::unordered_set<Node*>& visited) const
{
    if (pattern_idx < 0 || pattern_idx >= static_cast<int>(pattern_.nodes().size())) {
        return false;
    }

    const PatternNode& pnode = pattern_.nodes()[pattern_idx];

    // Check if this node matches the pattern node
    if (!node_matches(*graph_node, pnode)) {
        return pnode.optional;
    }

    // Check if already visited (avoid cycles)
    if (visited.count(graph_node)) {
        return false;
    }

    // Mark as visited
    visited.insert(graph_node);

    // Ensure nodes vector is large enough
    while (result.nodes.size() <= static_cast<size_t>(pattern_idx)) {
        result.nodes.push_back(nullptr);
    }
    result.nodes[pattern_idx] = graph_node;

    // Add to captures
    if (!pnode.capture.empty()) {
        result.captures[pnode.capture] = graph_node;
    }

    // Match edges (find input nodes)
    for (const auto& edge : pattern_.edges()) {
        if (edge.to_node != pattern_idx) continue;

        // Get the input at specified index
        if (edge.to_input >= static_cast<int>(graph_node->inputs().size())) {
            visited.erase(graph_node);
            return pnode.optional;
        }

        const std::string& input_name = graph_node->inputs()[edge.to_input];

        // Find the node that produces this input
        const std::string* producer_name = graph.get_producer(input_name);
        if (!producer_name) {
            // Input is a graph input or initializer, not from a node
            const PatternNode& expected = pattern_.nodes()[edge.from_node];
            if (!expected.optional) {
                visited.erase(graph_node);
                return false;
            }
            continue;
        }

        Node* input_node = graph.get_node(*producer_name);
        if (!input_node) {
            const PatternNode& expected = pattern_.nodes()[edge.from_node];
            if (!expected.optional) {
                visited.erase(graph_node);
                return false;
            }
            continue;
        }

        // Recursively match the input node
        if (!match_node(graph, input_node, edge.from_node, result, visited)) {
            visited.erase(graph_node);
            return false;
        }
    }

    return true;
}

std::optional<PatternMatch> PatternMatcher::match_at(
    const Graph& graph, Node* start) const
{
    if (pattern_.root() < 0) {
        return std::nullopt;
    }

    PatternMatch result;
    result.root = start;
    std::unordered_set<Node*> visited;

    if (match_node(graph, start, pattern_.root(), result, visited)) {
        return result;
    }

    return std::nullopt;
}

std::optional<PatternMatch> PatternMatcher::match_first(const Graph& graph) const {
    for (const auto& node : graph.nodes()) {
        auto match = match_at(graph, node.get());
        if (match) {
            return match;
        }
    }
    return std::nullopt;
}

std::vector<PatternMatch> PatternMatcher::match(const Graph& graph) const {
    std::vector<PatternMatch> matches;
    std::unordered_set<Node*> used_nodes;

    for (const auto& node : graph.nodes()) {
        // Skip if this node was already part of a match
        if (used_nodes.count(node.get())) continue;

        auto match = match_at(graph, node.get());
        if (match) {
            matches.push_back(*match);

            // Mark all matched nodes as used to avoid overlapping matches
            for (Node* n : match->nodes) {
                if (n) used_nodes.insert(n);
            }
        }
    }

    return matches;
}

} // namespace opt
} // namespace pyflame_rt
