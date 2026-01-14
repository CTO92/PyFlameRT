#include "pyflame_rt/quantization/quantizer.hpp"
#include "pyflame_rt/quantization/quant_ops.hpp"
#include <algorithm>
#include <stdexcept>

namespace pyflame_rt {
namespace quantization {

// ============================================================================
// Operator Classification
// ============================================================================

namespace {

// Operators that support INT8 quantization
const std::unordered_set<std::string> QUANTIZABLE_OPS = {
    "MatMul", "Gemm", "Conv", "Add", "Mul",
    "Relu", "MaxPool", "AveragePool", "GlobalAveragePool"
};

// Operators that must remain in float precision for numerical stability
const std::unordered_set<std::string> FLOAT_ONLY_OPS = {
    "Softmax", "LogSoftmax", "LayerNormalization", "BatchNormalization",
    "Div", "Sqrt", "Exp", "Log", "Pow", "ReduceMean", "ReduceSum"
};

// Map from original op to quantized op
const std::unordered_map<std::string, std::string> QUANTIZED_OP_MAP = {
    {"MatMul", "QuantizedMatMul"},
    {"Gemm", "QuantizedGemm"},
    {"Conv", "QuantizedConv"},
    {"Add", "QuantizedAdd"},
    {"Mul", "QuantizedMul"}
};

} // anonymous namespace

bool is_int8_quantizable(const std::string& op_type) {
    return QUANTIZABLE_OPS.find(op_type) != QUANTIZABLE_OPS.end();
}

bool requires_float_precision(const std::string& op_type) {
    return FLOAT_ONLY_OPS.find(op_type) != FLOAT_ONLY_OPS.end();
}

std::string get_quantized_op_type(const std::string& op_type) {
    auto it = QUANTIZED_OP_MAP.find(op_type);
    return (it != QUANTIZED_OP_MAP.end()) ? it->second : op_type;
}

// ============================================================================
// Quantizer Implementation
// ============================================================================

Quantizer::Quantizer(const QuantConfig& config)
    : config_(config)
{
}

Quantizer::~Quantizer() = default;

bool Quantizer::supports_quantization(const std::string& op_type) const {
    if (config_.exclude_ops.count(op_type) > 0) {
        return false;
    }
    if (requires_float_precision(op_type)) {
        return false;
    }
    return is_int8_quantizable(op_type);
}

std::vector<std::string> Quantizer::get_quantizable_ops() const {
    std::vector<std::string> result;
    for (const auto& op : QUANTIZABLE_OPS) {
        if (!is_excluded(op, "")) {
            result.push_back(op);
        }
    }
    return result;
}

bool Quantizer::is_excluded(const std::string& op_type,
                             const std::string& node_name) const {
    if (config_.exclude_ops.count(op_type) > 0) {
        return true;
    }
    if (!node_name.empty() && config_.exclude_nodes.count(node_name) > 0) {
        return true;
    }
    return false;
}

// ============================================================================
// FP16 Conversion
// ============================================================================

QuantizationResult Quantizer::convert_to_fp16(const Graph& graph) {
    QuantizationResult result;

    try {
        // Create a copy of the graph
        result.quantized_graph = std::make_unique<Graph>(graph.name());

        // Copy graph structure
        for (const auto& input : graph.inputs()) {
            result.quantized_graph->add_input(input);
        }
        for (const auto& output : graph.outputs()) {
            result.quantized_graph->add_output(output);
        }

        // Copy and convert initializers to FP16
        for (const auto& [name, tensor] : graph.initializers()) {
            if (tensor.dtype() == DType::Float32) {
                result.stats.original_size_bytes += tensor.size_bytes();
                Tensor fp16_tensor = cast_to_fp16(tensor);
                result.stats.quantized_size_bytes += fp16_tensor.size_bytes();
                result.quantized_graph->add_initializer(name, std::move(fp16_tensor));
                result.stats.weights_quantized++;
            } else {
                result.quantized_graph->add_initializer(name, tensor.clone());
            }
        }

        // Copy nodes - insert cast ops at boundaries if needed
        for (const auto& node : graph.nodes()) {
            result.quantized_graph->add_node(
                std::make_shared<Node>(*node));
            result.stats.nodes_quantized++;
        }

        result.quant_info.mode = QuantMode::FP16;
        result.quant_info.weights_quantized = true;
        result.success = true;

    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = std::string("FP16 conversion failed: ") + e.what();
    }

    return result;
}

// ============================================================================
// BFloat16 Conversion
// ============================================================================

QuantizationResult Quantizer::convert_to_bfloat16(const Graph& graph) {
    QuantizationResult result;

    try {
        result.quantized_graph = std::make_unique<Graph>(graph.name());

        // Copy graph structure
        for (const auto& input : graph.inputs()) {
            result.quantized_graph->add_input(input);
        }
        for (const auto& output : graph.outputs()) {
            result.quantized_graph->add_output(output);
        }

        // Copy and convert initializers to BFloat16
        for (const auto& [name, tensor] : graph.initializers()) {
            if (tensor.dtype() == DType::Float32) {
                result.stats.original_size_bytes += tensor.size_bytes();
                Tensor bf16_tensor = cast_to_bfloat16(tensor);
                result.stats.quantized_size_bytes += bf16_tensor.size_bytes();
                result.quantized_graph->add_initializer(name, std::move(bf16_tensor));
                result.stats.weights_quantized++;
            } else {
                result.quantized_graph->add_initializer(name, tensor.clone());
            }
        }

        // Copy nodes
        for (const auto& node : graph.nodes()) {
            result.quantized_graph->add_node(
                std::make_shared<Node>(*node));
            result.stats.nodes_quantized++;
        }

        result.quant_info.mode = QuantMode::BFloat16;
        result.quant_info.weights_quantized = true;
        result.success = true;

    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = std::string("BFloat16 conversion failed: ") + e.what();
    }

    return result;
}

// ============================================================================
// Dynamic INT8 Quantization
// ============================================================================

QuantizationResult Quantizer::quantize_dynamic(const Graph& graph) {
    QuantizationResult result;

    try {
        result.quantized_graph = std::make_unique<Graph>(graph.name());

        // Copy graph structure
        for (const auto& input : graph.inputs()) {
            result.quantized_graph->add_input(input);
        }
        for (const auto& output : graph.outputs()) {
            result.quantized_graph->add_output(output);
        }

        // Quantize weights
        for (const auto& [name, tensor] : graph.initializers()) {
            if (tensor.dtype() == DType::Float32) {
                result.stats.original_size_bytes += tensor.size_bytes();

                // Compute dynamic params and quantize
                QuantParams params = compute_dynamic_params(
                    tensor, config_.weight_dtype, config_.symmetric);
                QuantizedTensor qtensor(tensor, params);

                result.quantized_graph->add_initializer(name, qtensor.data().clone());
                result.quant_info.set_params(name, params);

                result.stats.quantized_size_bytes += qtensor.data().size_bytes();
                result.stats.weights_quantized++;
            } else {
                result.quantized_graph->add_initializer(name, tensor.clone());
            }
        }

        // Transform nodes to use quantized operators
        for (const auto& node : graph.nodes()) {
            auto new_node = std::make_shared<Node>(*node);

            if (supports_quantization(node->op_type()) &&
                !is_excluded(node->op_type(), node->name())) {
                transform_node_to_quantized(*new_node, result.quant_info);
                result.stats.nodes_quantized++;
            } else {
                result.stats.nodes_skipped++;
            }

            result.quantized_graph->add_node(new_node);
        }

        result.quant_info.mode = QuantMode::DynamicInt8;
        result.quant_info.weights_quantized = true;
        result.success = true;

    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = std::string("Dynamic quantization failed: ") + e.what();
    }

    return result;
}

// ============================================================================
// Static INT8 Quantization (with pre-computed quant info)
// ============================================================================

QuantizationResult Quantizer::quantize(const Graph& graph,
                                        const GraphQuantInfo& quant_info) {
    QuantizationResult result;
    result.quant_info = quant_info;

    try {
        result.quantized_graph = std::make_unique<Graph>(graph.name());

        // Copy graph structure
        for (const auto& input : graph.inputs()) {
            result.quantized_graph->add_input(input);
        }
        for (const auto& output : graph.outputs()) {
            result.quantized_graph->add_output(output);
        }

        // Quantize weights using provided params
        for (const auto& [name, tensor] : graph.initializers()) {
            const QuantParams* params = quant_info.get_params(name);

            if (params && tensor.dtype() == DType::Float32) {
                result.stats.original_size_bytes += tensor.size_bytes();

                QuantizedTensor qtensor(tensor, *params);
                result.quantized_graph->add_initializer(name, qtensor.data().clone());

                result.stats.quantized_size_bytes += qtensor.data().size_bytes();
                result.stats.weights_quantized++;
            } else {
                result.quantized_graph->add_initializer(name, tensor.clone());
            }
        }

        // Transform nodes
        for (const auto& node : graph.nodes()) {
            auto new_node = std::make_shared<Node>(*node);

            if (supports_quantization(node->op_type()) &&
                !is_excluded(node->op_type(), node->name())) {
                transform_node_to_quantized(*new_node, result.quant_info);
                result.stats.nodes_quantized++;
            } else {
                result.stats.nodes_skipped++;
            }

            result.quantized_graph->add_node(new_node);
        }

        result.success = true;

    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = std::string("Static quantization failed: ") + e.what();
    }

    return result;
}

// ============================================================================
// Quantization with Calibration
// ============================================================================

QuantizationResult Quantizer::quantize_with_calibration(
    const Graph& graph,
    CalibrationDataProvider data_provider,
    size_t num_batches)
{
    // Create calibrator and run calibration
    Calibrator calibrator(graph, config_);
    calibrator.calibrate(data_provider, num_batches);

    // Use calibrated params for quantization
    return quantize(graph, calibrator.quant_info());
}

// ============================================================================
// Internal Transformation Methods
// ============================================================================

void Quantizer::transform_node_to_quantized(Node& node,
                                             const GraphQuantInfo& info) {
    const std::string& op_type = node.op_type();

    // Update op type if there's a quantized version
    std::string quantized_op = get_quantized_op_type(op_type);
    if (quantized_op != op_type) {
        node.set_op_type(quantized_op);
    }

    // Add scale/zero-point attributes for inputs
    const auto& inputs = node.inputs();
    // Security fix MED-Q04: Limit attribute prefix indices to prevent overflow
    static const char* prefixes[] = {"a", "b", "c", "d", "e", "f", "g", "h"};
    static constexpr size_t max_prefixes = sizeof(prefixes) / sizeof(prefixes[0]);

    for (size_t i = 0; i < inputs.size() && i < max_prefixes; ++i) {
        const std::string& input_name = inputs[i];
        // Security fix MED-Q04/HIGH-Q05: Validate pointer before use
        const QuantParams* params = info.get_params(input_name);

        if (params != nullptr && params->is_valid() && !params->scales.empty()) {
            std::string prefix = prefixes[i];
            // Security fix MED-Q05: Bounds-checked access to scales and zero_points
            node.set_attr(prefix + "_scale", params->scales[0]);
            node.set_attr(prefix + "_zero_point",
                          static_cast<int64_t>(params->zero_points[0]));
        }
    }
}

void Quantizer::quantize_weights(Graph& graph, GraphQuantInfo& info,
                                  QuantizationResult::Stats& stats) {
    for (auto& [name, tensor] : graph.initializers()) {
        if (tensor.dtype() == DType::Float32) {
            stats.original_size_bytes += tensor.size_bytes();

            // Compute or get params
            QuantParams params;
            const QuantParams* existing = info.get_params(name);
            if (existing) {
                params = *existing;
            } else {
                params = compute_dynamic_params(tensor,
                    config_.weight_dtype, config_.symmetric);
                info.set_params(name, params);
            }

            // Quantize
            QuantizedTensor qtensor(tensor, params);
            tensor = qtensor.data().clone();

            stats.quantized_size_bytes += tensor.size_bytes();
            stats.weights_quantized++;
        }
    }
}

void Quantizer::convert_weights_to_fp16(Graph& graph,
                                         QuantizationResult::Stats& stats) {
    for (auto& [name, tensor] : graph.initializers()) {
        if (tensor.dtype() == DType::Float32) {
            stats.original_size_bytes += tensor.size_bytes();
            tensor = cast_to_fp16(tensor);
            stats.quantized_size_bytes += tensor.size_bytes();
            stats.weights_quantized++;
        }
    }
}

void Quantizer::convert_weights_to_bf16(Graph& graph,
                                         QuantizationResult::Stats& stats) {
    for (auto& [name, tensor] : graph.initializers()) {
        if (tensor.dtype() == DType::Float32) {
            stats.original_size_bytes += tensor.size_bytes();
            tensor = cast_to_bfloat16(tensor);
            stats.quantized_size_bytes += tensor.size_bytes();
            stats.weights_quantized++;
        }
    }
}

void Quantizer::insert_quantize_ops(Graph& /*graph*/,
                                     const GraphQuantInfo& /*info*/) {
    // Insert Quantize nodes before quantized ops for non-quantized inputs
    // This is needed when activations come from non-quantized ops
    // For now, skip as we handle this at runtime
}

void Quantizer::insert_dequantize_ops(Graph& /*graph*/,
                                       const GraphQuantInfo& /*info*/) {
    // Insert Dequantize nodes after quantized ops if outputs need float
    // For now, skip as we output float from quantized ops
}

} // namespace quantization
} // namespace pyflame_rt
