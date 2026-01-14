#include "pyflame_rt/distillation/distillation.hpp"
#include <algorithm>
#include <cmath>

namespace pyflame_rt {
namespace distillation {

Graph create_student_architecture(
    const Graph& teacher,
    const StudentArchitectureConfig& config)
{
    // Create a compressed student model based on teacher architecture
    // This is a simplified implementation that:
    // 1. Copies the graph structure
    // 2. Reduces hidden dimensions by compression_ratio
    // 3. Optionally removes layers

    Graph student = teacher.clone();

    // Get all nodes
    auto nodes = student.nodes();

    // Find layers to compress (Conv, Linear, etc.)
    std::vector<std::shared_ptr<Node>> compressible_nodes;
    for (const auto& node : nodes) {
        std::string op_type = node->op_type();
        if (op_type == "Conv" || op_type == "Gemm" || op_type == "MatMul" ||
            op_type == "Linear" || op_type == "ConvTranspose") {
            compressible_nodes.push_back(node);
        }
    }

    if (compressible_nodes.empty()) {
        return student;  // Nothing to compress
    }

    // Determine which nodes to skip for compression (preserve endpoints)
    size_t start_idx = 0;
    size_t end_idx = compressible_nodes.size();

    if (config.preserve_endpoints && compressible_nodes.size() > 2) {
        start_idx = 1;
        end_idx = compressible_nodes.size() - 1;
    }

    // Compress each layer's weights
    for (size_t i = start_idx; i < end_idx; ++i) {
        auto& node = compressible_nodes[i];

        // Find weight initializer for this node
        if (node->inputs().size() < 2) continue;

        std::string weight_name = node->inputs()[1];
        Tensor* weights = student.get_mutable_initializer(weight_name);
        if (!weights) continue;

        auto shape = weights->shape();
        if (shape.size() < 2) continue;

        // Calculate new dimensions
        std::vector<int64_t> new_shape = shape;

        if (config.uniform_compression) {
            // Compress all dimensions uniformly
            for (size_t d = 0; d < new_shape.size(); ++d) {
                int64_t new_dim = static_cast<int64_t>(
                    std::ceil(shape[d] * config.compression_ratio));
                new_dim = std::max(new_dim, static_cast<int64_t>(config.min_width));
                new_shape[d] = new_dim;
            }
        } else {
            // Only compress output dimension (first dimension for most ops)
            int64_t new_out = static_cast<int64_t>(
                std::ceil(shape[0] * config.compression_ratio));
            new_out = std::max(new_out, static_cast<int64_t>(config.min_width));
            new_shape[0] = new_out;
        }

        // Create new compressed weight tensor
        // We'll use simple truncation - a real implementation would use
        // importance-based selection or random initialization
        Tensor new_weights(new_shape, weights->dtype());
        float* dst = static_cast<float*>(new_weights.data());
        const float* src = static_cast<const float*>(weights->data());

        // Security: null pointer checks (HIGH-D7 fix)
        if (!dst || !src) {
            continue;  // Skip this layer if pointers are null
        }

        // Copy data (truncated)
        size_t new_elements = new_weights.num_elements();
        size_t old_elements = weights->num_elements();

        // Simple copy for now - real implementation would be more sophisticated
        size_t copy_elements = std::min(new_elements, old_elements);
        std::copy(src, src + copy_elements, dst);

        // Zero-initialize any remaining elements
        if (new_elements > copy_elements) {
            std::fill(dst + copy_elements, dst + new_elements, 0.0f);
        }

        // Update the weight tensor
        *weights = std::move(new_weights);

        // If node has bias, compress it too
        if (node->inputs().size() > 2) {
            std::string bias_name = node->inputs()[2];
            Tensor* bias = student.get_mutable_initializer(bias_name);
            if (bias && bias->shape().size() == 1) {
                int64_t new_bias_size = new_shape[0];
                std::vector<int64_t> new_bias_shape = {new_bias_size};

                Tensor new_bias(new_bias_shape, bias->dtype());
                float* bias_dst = static_cast<float*>(new_bias.data());
                const float* bias_src = static_cast<const float*>(bias->data());

                // Security: null pointer checks (HIGH-D7 fix)
                if (!bias_dst || !bias_src) {
                    continue;  // Skip bias compression if pointers are null
                }

                size_t bias_copy = std::min(
                    static_cast<size_t>(new_bias_size),
                    bias->num_elements());
                std::copy(bias_src, bias_src + bias_copy, bias_dst);

                if (static_cast<size_t>(new_bias_size) > bias_copy) {
                    std::fill(bias_dst + bias_copy, bias_dst + new_bias_size, 0.0f);
                }

                *bias = std::move(new_bias);
            }
        }
    }

    // Handle layer reduction if specified
    if (config.reduce_layers > 0 && compressible_nodes.size() > config.reduce_layers + 2) {
        // This is complex as it requires rewiring the graph
        // For now, we skip this feature
        // A real implementation would:
        // 1. Identify layers to remove
        // 2. Add skip connections if needed
        // 3. Update input/output connections
    }

    return student;
}

} // namespace distillation
} // namespace pyflame_rt
