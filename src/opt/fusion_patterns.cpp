#include "pyflame_rt/opt/operator_fusion.hpp"

namespace pyflame_rt {
namespace opt {

namespace {

// ============================================================================
// Helper Functions
// ============================================================================

/// Check if a node has exactly one consumer
bool has_single_consumer(const Graph& graph, const Node& node) {
    if (node.outputs().empty()) return false;

    const std::string& output = node.outputs()[0];
    int consumer_count = 0;

    for (const auto& other : graph.nodes()) {
        for (const auto& input : other->inputs()) {
            if (input == output) {
                consumer_count++;
                if (consumer_count > 1) return false;
            }
        }
    }

    return consumer_count == 1;
}

/// Generate unique name for fused node
std::string make_fused_name(const PatternMatch& match, const std::string& suffix) {
    if (match.root) {
        return match.root->name() + "_fused_" + suffix;
    }
    return "fused_" + suffix;
}

// ============================================================================
// Fusion: Add + ReLU -> FusedAddRelu
// ============================================================================

FusionPattern create_add_relu_pattern() {
    FusionPattern pattern;
    pattern.name = "AddRelu";
    pattern.fused_op = "FusedAddRelu";

    // Pattern: Add -> Relu
    Pattern p;
    int add_idx = p.add_node(PatternNode::op("Add", "add"));
    int relu_idx = p.add_node(PatternNode::op("Relu", "relu"));
    p.add_edge(add_idx, relu_idx);  // Add output -> Relu input
    p.set_root(relu_idx);
    pattern.pattern = p;

    // Constraint: Add must have single consumer
    pattern.constraint = [](const PatternMatch& match, const Graph& graph) {
        Node* add = match.captures.at("add");
        return has_single_consumer(graph, *add);
    };

    // Fusion function
    pattern.fuse_func = [](const PatternMatch& match, Graph& graph) {
        Node* add = match.captures.at("add");
        Node* relu = match.captures.at("relu");

        auto fused = std::make_shared<Node>(
            make_fused_name(match, "add_relu"),
            "FusedAddRelu",
            add->inputs(),
            relu->outputs()
        );

        return fused;
    };

    return pattern;
}

// ============================================================================
// Fusion: Mul + Add -> FMA (Fused Multiply-Add)
// ============================================================================

FusionPattern create_mul_add_pattern() {
    FusionPattern pattern;
    pattern.name = "MulAdd";
    pattern.fused_op = "FMA";

    // Pattern: Mul -> Add
    Pattern p;
    int mul_idx = p.add_node(PatternNode::op("Mul", "mul"));
    int add_idx = p.add_node(PatternNode::op("Add", "add"));
    p.add_edge(mul_idx, add_idx);
    p.set_root(add_idx);
    pattern.pattern = p;

    // Constraint: Mul must have single consumer
    pattern.constraint = [](const PatternMatch& match, const Graph& graph) {
        Node* mul = match.captures.at("mul");
        return has_single_consumer(graph, *mul);
    };

    // Fusion function
    pattern.fuse_func = [](const PatternMatch& match, Graph& graph) {
        Node* mul = match.captures.at("mul");
        Node* add = match.captures.at("add");

        // FMA: a * b + c
        // Mul inputs are a, b; Add has Mul output and c
        std::vector<std::string> inputs = mul->inputs();

        // Find the other input to Add (the one that's not from Mul)
        for (const auto& inp : add->inputs()) {
            if (inp != mul->outputs()[0]) {
                inputs.push_back(inp);
                break;
            }
        }

        auto fused = std::make_shared<Node>(
            make_fused_name(match, "fma"),
            "FMA",
            inputs,
            add->outputs()
        );

        return fused;
    };

    return pattern;
}

// ============================================================================
// Fusion: MatMul + Add -> Gemm
// ============================================================================

FusionPattern create_matmul_add_pattern() {
    FusionPattern pattern;
    pattern.name = "MatMulAdd";
    pattern.fused_op = "Gemm";

    // Pattern: MatMul -> Add
    Pattern p;
    int matmul_idx = p.add_node(PatternNode::op("MatMul", "matmul"));
    int add_idx = p.add_node(PatternNode::op("Add", "add"));
    p.add_edge(matmul_idx, add_idx);
    p.set_root(add_idx);
    pattern.pattern = p;

    // Constraint: MatMul must have single consumer
    pattern.constraint = [](const PatternMatch& match, const Graph& graph) {
        Node* matmul = match.captures.at("matmul");
        return has_single_consumer(graph, *matmul);
    };

    // Fusion function
    pattern.fuse_func = [](const PatternMatch& match, Graph& graph) {
        Node* matmul = match.captures.at("matmul");
        Node* add = match.captures.at("add");

        // Gemm: alpha * A * B + beta * C
        std::vector<std::string> inputs = matmul->inputs();

        // Find the bias input to Add
        for (const auto& inp : add->inputs()) {
            if (inp != matmul->outputs()[0]) {
                inputs.push_back(inp);
                break;
            }
        }

        auto fused = std::make_shared<Node>(
            make_fused_name(match, "gemm"),
            "Gemm",
            inputs,
            add->outputs()
        );

        // Default Gemm attributes
        fused->set_attr("alpha", 1.0f);
        fused->set_attr("beta", 1.0f);
        fused->set_attr("transA", int64_t{0});
        fused->set_attr("transB", int64_t{0});

        return fused;
    };

    return pattern;
}

// ============================================================================
// Fusion: Conv + Relu -> FusedConvRelu
// ============================================================================

FusionPattern create_conv_relu_pattern() {
    FusionPattern pattern;
    pattern.name = "ConvRelu";
    pattern.fused_op = "FusedConvRelu";

    // Pattern: Conv -> Relu
    Pattern p;
    int conv_idx = p.add_node(PatternNode::op("Conv", "conv"));
    int relu_idx = p.add_node(PatternNode::op("Relu", "relu"));
    p.add_edge(conv_idx, relu_idx);
    p.set_root(relu_idx);
    pattern.pattern = p;

    // Constraint: Conv must have single consumer
    pattern.constraint = [](const PatternMatch& match, const Graph& graph) {
        Node* conv = match.captures.at("conv");
        return has_single_consumer(graph, *conv);
    };

    // Fusion function
    pattern.fuse_func = [](const PatternMatch& match, Graph& graph) {
        Node* conv = match.captures.at("conv");
        Node* relu = match.captures.at("relu");

        auto fused = std::make_shared<Node>(
            make_fused_name(match, "conv_relu"),
            "FusedConvRelu",
            conv->inputs(),
            relu->outputs()
        );

        // Copy Conv attributes
        for (const auto& attr_name : conv->attr_names()) {
            // Copy attribute (simplified - would need type handling)
        }

        return fused;
    };

    return pattern;
}

// ============================================================================
// Fusion: BatchNorm + Relu -> FusedBNRelu
// ============================================================================

FusionPattern create_bn_relu_pattern() {
    FusionPattern pattern;
    pattern.name = "BNRelu";
    pattern.fused_op = "FusedBNRelu";

    // Pattern: BatchNormalization -> Relu
    Pattern p;
    int bn_idx = p.add_node(PatternNode::op("BatchNormalization", "bn"));
    int relu_idx = p.add_node(PatternNode::op("Relu", "relu"));
    p.add_edge(bn_idx, relu_idx);
    p.set_root(relu_idx);
    pattern.pattern = p;

    // Constraint: BN must have single consumer
    pattern.constraint = [](const PatternMatch& match, const Graph& graph) {
        Node* bn = match.captures.at("bn");
        return has_single_consumer(graph, *bn);
    };

    // Fusion function
    pattern.fuse_func = [](const PatternMatch& match, Graph& graph) {
        Node* bn = match.captures.at("bn");
        Node* relu = match.captures.at("relu");

        auto fused = std::make_shared<Node>(
            make_fused_name(match, "bn_relu"),
            "FusedBNRelu",
            bn->inputs(),
            relu->outputs()
        );

        return fused;
    };

    return pattern;
}

} // anonymous namespace

// ============================================================================
// Register Built-in Patterns
// ============================================================================

void OperatorFusionPass::register_builtin_patterns() {
    // Element-wise + Activation
    if (config_.fuse_elementwise_activation) {
        register_pattern(create_add_relu_pattern());
        register_pattern(create_mul_add_pattern());
    }

    // MatMul + Bias
    if (config_.fuse_matmul_add) {
        register_pattern(create_matmul_add_pattern());
    }

    // Conv + Activation
    if (config_.fuse_conv_bn_activation) {
        register_pattern(create_conv_relu_pattern());
        register_pattern(create_bn_relu_pattern());
    }
}

} // namespace opt
} // namespace pyflame_rt
