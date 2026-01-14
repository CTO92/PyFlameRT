#pragma once

#include "pyflame_rt/graph.hpp"
#include "pyflame_rt/backend.hpp"
#include "pyflame_rt/options.hpp"
#include "pyflame_rt/types.hpp"
#include "pyflame_rt/quantization/quant_params.hpp"
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <optional>

namespace pyflame_rt {

/// Report on quantization applied to a model
struct QuantizationReport {
    quantization::QuantMode mode = quantization::QuantMode::None;
    size_t nodes_quantized = 0;
    size_t nodes_total = 0;
    float compression_ratio = 1.0f;
    float original_size_mb = 0.0f;
    float quantized_size_mb = 0.0f;
    bool weights_quantized = false;
    bool activations_quantized = false;
};

/// High-level inference session for model execution
/// Provides an ONNX Runtime-compatible API
class InferenceSession {
public:
    /// Load a model and prepare for inference
    /// @param model_path Path to model file (.pfm)
    /// @param options Session configuration options
    /// @param providers Execution provider priority list (optional)
    explicit InferenceSession(
        const std::string& model_path,
        SessionOptions options = {},
        std::vector<std::string> providers = {}
    );

    ~InferenceSession();

    // Non-copyable
    InferenceSession(const InferenceSession&) = delete;
    InferenceSession& operator=(const InferenceSession&) = delete;

    // Movable
    InferenceSession(InferenceSession&&) noexcept;
    InferenceSession& operator=(InferenceSession&&) noexcept;

    /// Run inference on the model
    /// @param output_names Names of outputs to return (empty = all outputs)
    /// @param input_feed Dictionary mapping input names to tensors
    /// @param run_options Per-run configuration (optional)
    /// @return List of output tensors
    std::vector<Tensor> run(
        const std::vector<std::string>& output_names,
        const std::unordered_map<std::string, Tensor>& input_feed,
        const RunOptions& run_options = {}
    );

    /// Get input tensor metadata
    std::vector<NodeArg> get_inputs() const;

    /// Get output tensor metadata
    std::vector<NodeArg> get_outputs() const;

    /// Get model metadata
    ModelMetadata get_modelmeta() const;

    /// Get list of available execution providers
    std::vector<std::string> get_providers() const;

    /// Access underlying graph (advanced use)
    const Graph& graph() const { return *graph_; }

    // ========================================================================
    // Quantization (Phase 4)
    // ========================================================================

    /// Check if session uses quantization
    bool is_quantized() const { return quant_info_.has_value(); }

    /// Get quantization report
    QuantizationReport quantization_report() const;

    /// Get quantization info (if quantized)
    const quantization::GraphQuantInfo* quant_info() const {
        return quant_info_.has_value() ? &quant_info_.value() : nullptr;
    }

private:
    void validate_options();
    void select_backend(const std::vector<std::string>& providers);
    void optimize_graph();
    void apply_quantization();
    void validate_inputs(const std::unordered_map<std::string, Tensor>& input_feed) const;

    SessionOptions options_;
    std::unique_ptr<Graph> graph_;
    std::unique_ptr<Backend> backend_;
    ModelMetadata metadata_;

    // Quantization state
    std::optional<quantization::GraphQuantInfo> quant_info_;
    QuantizationReport quant_report_;
};

} // namespace pyflame_rt
