#pragma once

#include "pyflame_rt/quantization/quant_config.hpp"
#include "pyflame_rt/quantization/quant_params.hpp"
#include "pyflame_rt/quantization/calibrator.hpp"
#include "pyflame_rt/graph.hpp"
#include <memory>
#include <string>
#include <vector>
#include <unordered_set>

namespace pyflame_rt {
namespace quantization {

/// Result of quantization transformation
struct QuantizationResult {
    /// The quantized graph (or nullptr on failure)
    std::unique_ptr<Graph> quantized_graph;

    /// Quantization info for all tensors
    GraphQuantInfo quant_info;

    /// Statistics about the quantization
    struct Stats {
        size_t nodes_quantized = 0;
        size_t nodes_skipped = 0;
        size_t weights_quantized = 0;
        size_t original_size_bytes = 0;
        size_t quantized_size_bytes = 0;

        float compression_ratio() const {
            return quantized_size_bytes > 0
                ? static_cast<float>(original_size_bytes) / quantized_size_bytes
                : 1.0f;
        }

        float original_size_mb() const {
            return static_cast<float>(original_size_bytes) / (1024.0f * 1024.0f);
        }

        float quantized_size_mb() const {
            return static_cast<float>(quantized_size_bytes) / (1024.0f * 1024.0f);
        }
    } stats;

    /// Whether quantization succeeded
    bool success = false;

    /// Error message if quantization failed
    std::string error_message;
};

/// Graph quantizer - transforms graphs for quantized inference
class Quantizer {
public:
    /// Create quantizer with configuration
    explicit Quantizer(const QuantConfig& config);
    ~Quantizer();

    // ========================================================================
    // Quantization Methods
    // ========================================================================

    /// Quantize graph with pre-computed quantization info
    /// Use this for static quantization after calibration
    QuantizationResult quantize(const Graph& graph,
                                 const GraphQuantInfo& quant_info);

    /// Quantize graph with dynamic quantization (no calibration needed)
    QuantizationResult quantize_dynamic(const Graph& graph);

    /// Convert graph to FP16
    QuantizationResult convert_to_fp16(const Graph& graph);

    /// Convert graph to BFloat16
    QuantizationResult convert_to_bfloat16(const Graph& graph);

    /// Quantize graph with calibration data
    /// Performs calibration and then quantization
    QuantizationResult quantize_with_calibration(
        const Graph& graph,
        CalibrationDataProvider data_provider,
        size_t num_batches);

    // ========================================================================
    // Query Methods
    // ========================================================================

    /// Check if operator supports quantization
    bool supports_quantization(const std::string& op_type) const;

    /// Get list of quantizable operators
    std::vector<std::string> get_quantizable_ops() const;

    /// Check if operator should be excluded from quantization
    bool is_excluded(const std::string& op_type,
                      const std::string& node_name) const;

    /// Get configuration
    const QuantConfig& config() const { return config_; }

private:
    QuantConfig config_;

    // Internal transformation methods
    void transform_node_to_quantized(Node& node, const GraphQuantInfo& info);
    void insert_quantize_ops(Graph& graph, const GraphQuantInfo& info);
    void insert_dequantize_ops(Graph& graph, const GraphQuantInfo& info);
    void quantize_weights(Graph& graph, GraphQuantInfo& info,
                          QuantizationResult::Stats& stats);
    void convert_weights_to_fp16(Graph& graph, QuantizationResult::Stats& stats);
    void convert_weights_to_bf16(Graph& graph, QuantizationResult::Stats& stats);
};

// ============================================================================
// Operators that support INT8 quantization
// ============================================================================

/// Check if an operator type supports INT8 quantization
bool is_int8_quantizable(const std::string& op_type);

/// Check if an operator type should remain in float precision
bool requires_float_precision(const std::string& op_type);

/// Get the quantized version of an operator type
std::string get_quantized_op_type(const std::string& op_type);

} // namespace quantization
} // namespace pyflame_rt
