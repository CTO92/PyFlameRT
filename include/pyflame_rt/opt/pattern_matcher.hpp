#pragma once

#include "pyflame_rt/graph.hpp"
#include "pyflame_rt/node.hpp"

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace pyflame_rt {
namespace opt {

/// Predicate for matching nodes
using NodePredicate = std::function<bool(const Node&)>;

/// Single node in a pattern
struct PatternNode {
    /// Operator type to match (empty = wildcard)
    std::string op_type;

    /// Additional predicate (optional)
    NodePredicate predicate;

    /// Capture name for matched node
    std::string capture;

    /// Whether this is an optional match
    bool optional = false;

    /// Factory methods
    static PatternNode op(const std::string& op_type, const std::string& capture = "");
    static PatternNode any(const std::string& capture = "");
    static PatternNode matching(NodePredicate pred, const std::string& capture = "");
};

/// Edge in a pattern (connects pattern nodes)
struct PatternEdge {
    int from_node;      // Source pattern node index
    int from_output;    // Output index from source
    int to_node;        // Target pattern node index
    int to_input;       // Input index on target
};

/// Result of a pattern match
struct PatternMatch {
    /// Root node of the match
    Node* root = nullptr;

    /// All matched nodes in pattern order
    std::vector<Node*> nodes;

    /// Captured nodes by name
    std::unordered_map<std::string, Node*> captures;

    /// Get captured node (throws if not found)
    Node* get(const std::string& name) const;

    /// Check if capture exists
    bool has(const std::string& name) const;
};

/// Graph pattern for matching subgraphs
class Pattern {
public:
    Pattern() = default;

    /// Add a node to the pattern
    /// @return Index of the added node
    int add_node(PatternNode node);

    /// Add edge connecting pattern nodes
    void add_edge(int from, int from_output, int to, int to_input);
    void add_edge(int from, int to);  // Default: output 0 to input 0

    /// Set the root node (entry point for matching)
    void set_root(int node_idx);

    /// Get root node index
    int root() const { return root_; }

    /// Get pattern nodes
    const std::vector<PatternNode>& nodes() const { return nodes_; }

    /// Get pattern edges
    const std::vector<PatternEdge>& edges() const { return edges_; }

    // ========================================================================
    // Convenience builders
    // ========================================================================

    /// Create pattern matching a sequence of operators (from input to output)
    static Pattern sequence(const std::vector<std::string>& op_types);

    /// Create pattern for single operator
    static Pattern single(const std::string& op_type);

    /// Create pattern for operator with predicate
    static Pattern single(const std::string& op_type, NodePredicate pred);

private:
    std::vector<PatternNode> nodes_;
    std::vector<PatternEdge> edges_;
    int root_ = -1;
};

/// Pattern matcher for finding subgraphs
class PatternMatcher {
public:
    explicit PatternMatcher(const Pattern& pattern);

    /// Find all matches of pattern in graph
    std::vector<PatternMatch> match(const Graph& graph) const;

    /// Find first match (more efficient if only one needed)
    std::optional<PatternMatch> match_first(const Graph& graph) const;

    /// Check if pattern matches starting from a node
    std::optional<PatternMatch> match_at(const Graph& graph, Node* start) const;

private:
    const Pattern& pattern_;

    /// Recursive matching helper
    bool match_node(
        const Graph& graph,
        Node* graph_node,
        int pattern_idx,
        PatternMatch& result,
        std::unordered_set<Node*>& visited
    ) const;

    /// Check if node matches pattern node
    bool node_matches(const Node& node, const PatternNode& pattern) const;
};

} // namespace opt
} // namespace pyflame_rt
