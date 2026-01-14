#include "pyflame_rt/session.hpp"
#include "pyflame_rt/errors.hpp"
#include "pyflame_rt/opt/passes.hpp"
#include "pyflame_rt/quantization/quantizer.hpp"
#include "pyflame_rt/quantization/calibrator.hpp"
#include "io/loader.hpp"
#include "backends/cpu/executor.hpp"
#include <iostream>
#include <limits>
#include <unordered_set>
#include <chrono>

namespace pyflame_rt {

InferenceSession::InferenceSession(
    const std::string& model_path,
    SessionOptions options,
    std::vector<std::string> providers)
    : options_(std::move(options))
{
    validate_options();

    // Load model
    graph_ = load_model(model_path);

    // Validate graph
    auto errors = graph_->validate();
    if (!errors.empty()) {
        throw ValidationError(errors);
    }

    // Run optimization passes if enabled
    if (options_.optimization_level != OptLevel::None) {
        optimize_graph();
    }

    // Apply quantization if configured
    if (options_.quantization.has_value()) {
        apply_quantization();
    }

    // Select backend
    select_backend(providers);

    // Cache metadata
    metadata_.graph_name = graph_->name();
    metadata_.producer_name = "PyFlameRT";
    metadata_.producer_version = "0.1.0";
}

InferenceSession::~InferenceSession() = default;

InferenceSession::InferenceSession(InferenceSession&&) noexcept = default;
InferenceSession& InferenceSession::operator=(InferenceSession&&) noexcept = default;

void InferenceSession::validate_options() {
    auto errors = options_.validate();
    if (!errors.empty()) {
        std::string msg = "Invalid session options:\n";
        for (const auto& e : errors) {
            msg += "  - " + e + "\n";
        }
        throw PyFlameRTError(msg);
    }
}

void InferenceSession::select_backend(const std::vector<std::string>& /*providers*/) {
    // For Phase 1, only CPU backend is available
    if (options_.device != "cpu") {
        std::cerr << "Warning: Device '" << options_.device
                  << "' not available, falling back to CPU\n";
    }

    // Convert memory limit from MB to bytes with overflow protection (LOW-04 fix)
    std::optional<size_t> memory_limit_bytes;
    if (options_.memory_limit_mb.has_value() && options_.memory_limit_mb.value() > 0) {
        size_t limit_mb = static_cast<size_t>(options_.memory_limit_mb.value());
        // Check for overflow: limit_mb * 1024 * 1024 must not overflow size_t
        // Max safe value is SIZE_MAX / (1024 * 1024)
        constexpr size_t max_safe_mb = std::numeric_limits<size_t>::max() / (1024ULL * 1024ULL);
        if (limit_mb > max_safe_mb) {
            throw PyFlameRTError(
                "Memory limit too large: " + std::to_string(limit_mb) +
                " MB would cause overflow. Maximum is " +
                std::to_string(max_safe_mb) + " MB");
        }
        memory_limit_bytes = limit_mb * 1024ULL * 1024ULL;
    }

    backend_ = std::make_unique<CPUExecutor>(
        options_.num_threads,
        memory_limit_bytes,
        options_.strict_math_mode  // HIGH-04 fix
    );
}

std::vector<Tensor> InferenceSession::run(
    const std::vector<std::string>& output_names,
    const std::unordered_map<std::string, Tensor>& input_feed,
    const RunOptions& /*run_options*/)
{
    // Security fix HIGH-12/MED-01: Validate inputs before execution
    validate_inputs(input_feed);

    // Security: Validate output names exist
    if (!output_names.empty()) {
        std::unordered_set<std::string> valid_outputs;
        for (const auto& info : graph_->outputs()) {
            valid_outputs.insert(info.name);
        }
        for (const auto& name : output_names) {
            if (valid_outputs.find(name) == valid_outputs.end()) {
                throw PyFlameRTError(
                    "Requested output '" + name + "' does not exist in graph");
            }
        }
    }

    return backend_->execute(*graph_, input_feed, output_names);
}

void InferenceSession::validate_inputs(
    const std::unordered_map<std::string, Tensor>& input_feed) const
{
    // Build set of expected inputs
    std::unordered_map<std::string, const TensorInfo*> expected_inputs;
    for (const auto& info : graph_->inputs()) {
        expected_inputs[info.name] = &info;
    }

    // Check that all required inputs are provided
    for (const auto& [name, info] : expected_inputs) {
        // Skip if this is an initializer (weight/bias)
        if (graph_->has_initializer(name)) {
            continue;
        }

        auto it = input_feed.find(name);
        if (it == input_feed.end()) {
            throw PyFlameRTError(
                "Missing required input '" + name + "'");
        }

        // Validate dtype matches
        if (it->second.dtype() != info->dtype) {
            throw PyFlameRTError(
                "Input '" + name + "' has dtype " + dtype_name(it->second.dtype()) +
                " but expected " + dtype_name(info->dtype));
        }

        // Validate shape matches (if not dynamic)
        if (!info->is_dynamic()) {
            const auto& expected_shape = info->shape;
            const auto& actual_shape = it->second.shape();

            if (expected_shape.size() != actual_shape.size()) {
                throw PyFlameRTError(
                    "Input '" + name + "' has " + std::to_string(actual_shape.size()) +
                    " dimensions but expected " + std::to_string(expected_shape.size()));
            }

            for (size_t i = 0; i < expected_shape.size(); ++i) {
                if (expected_shape[i].has_value() &&
                    expected_shape[i].value() != actual_shape[i]) {
                    throw PyFlameRTError(
                        "Input '" + name + "' dimension " + std::to_string(i) +
                        " has size " + std::to_string(actual_shape[i]) +
                        " but expected " + std::to_string(expected_shape[i].value()));
                }
            }
        }
    }
}

std::vector<NodeArg> InferenceSession::get_inputs() const {
    std::vector<NodeArg> result;
    result.reserve(graph_->inputs().size());
    for (const auto& info : graph_->inputs()) {
        result.push_back(NodeArg::from_tensor_info(info));
    }
    return result;
}

std::vector<NodeArg> InferenceSession::get_outputs() const {
    std::vector<NodeArg> result;
    result.reserve(graph_->outputs().size());
    for (const auto& info : graph_->outputs()) {
        result.push_back(NodeArg::from_tensor_info(info));
    }
    return result;
}

ModelMetadata InferenceSession::get_modelmeta() const {
    return metadata_;
}

std::vector<std::string> InferenceSession::get_providers() const {
    return {"CPUExecutionProvider"};
}

void InferenceSession::optimize_graph() {
    // Create pass manager with configuration (MED-01: timeout is enforced by PassManager)
    opt::PassManagerConfig pm_config;
    pm_config.opt_level = options_.optimization_level;
    pm_config.verbose = options_.verbose_optimization;
    pm_config.validate_after_pass = true;
    pm_config.max_iterations = 100;  // Security: limit iterations to prevent infinite loops
    pm_config.timeout_ms = options_.optimization_timeout_ms;  // MED-01: enforce timeout

    // Create pass manager and register built-in passes
    opt::PassManager pm(pm_config);
    pm.register_pass(std::make_unique<opt::ConstantFoldingPass>());
    pm.register_pass(std::make_unique<opt::DeadCodeEliminationPass>());
    pm.register_pass(std::make_unique<opt::CSEPass>());
    pm.register_pass(std::make_unique<opt::OperatorFusionPass>());
    pm.register_pass(std::make_unique<opt::LayoutOptimizationPass>());

    // Run optimization until fixed point (MED-01: will abort on timeout)
    auto result = pm.run_until_fixed_point(*graph_);

    // MED-01: Check if optimization was aborted due to timeout
    if (result.timed_out) {
        std::cerr << "Warning: Optimization was aborted due to timeout. "
                  << "Consider using lower optimization level or increasing timeout.\n";
    }

    if (options_.verbose_optimization) {
        std::cerr << "Optimization completed:"
                  << " nodes_removed=" << result.stats.nodes_removed
                  << " nodes_added=" << result.stats.nodes_added
                  << " nodes_fused=" << result.stats.nodes_fused
                  << " constants_folded=" << result.stats.constants_folded
                  << (result.timed_out ? " (TIMED OUT)" : "")
                  << "\n";

        for (const auto& warning : result.warnings) {
            std::cerr << "Optimization warning: " << warning << "\n";
        }
    }

    // Re-validate after optimization (security: always validate)
    auto errors = graph_->validate();
    if (!errors.empty()) {
        throw ValidationError(errors);
    }
}

void InferenceSession::apply_quantization() {
    using namespace quantization;

    const QuantConfig& config = options_.quantization.value();

    // Validate config
    if (!config.is_valid()) {
        throw PyFlameRTError("Invalid quantization config: " +
                             config.validation_error());
    }

    Quantizer quantizer(config);
    QuantizationResult result;

    switch (config.mode) {
        case QuantMode::FP16:
            result = quantizer.convert_to_fp16(*graph_);
            break;

        case QuantMode::BFloat16:
            result = quantizer.convert_to_bfloat16(*graph_);
            break;

        case QuantMode::DynamicInt8:
            result = quantizer.quantize_dynamic(*graph_);
            break;

        case QuantMode::StaticInt8:
            if (options_.calibration_data) {
                // Run calibration
                result = quantizer.quantize_with_calibration(
                    *graph_,
                    options_.calibration_data,
                    options_.calibration_batches);
            } else {
                // Fall back to dynamic quantization
                if (options_.verbose_optimization) {
                    std::cerr << "Warning: Static INT8 requested but no "
                              << "calibration data provided, using dynamic "
                              << "quantization instead\n";
                }
                result = quantizer.quantize_dynamic(*graph_);
            }
            break;

        case QuantMode::None:
        default:
            return;  // No quantization
    }

    if (!result.success) {
        throw PyFlameRTError("Quantization failed: " + result.error_message);
    }

    // Replace graph with quantized version
    if (result.quantized_graph) {
        graph_ = std::move(result.quantized_graph);
    }

    // Store quantization info
    quant_info_ = std::move(result.quant_info);

    // Update report
    quant_report_.mode = config.mode;
    quant_report_.nodes_quantized = result.stats.nodes_quantized;
    quant_report_.nodes_total = result.stats.nodes_quantized + result.stats.nodes_skipped;
    quant_report_.compression_ratio = result.stats.compression_ratio();
    quant_report_.original_size_mb = result.stats.original_size_mb();
    quant_report_.quantized_size_mb = result.stats.quantized_size_mb();
    quant_report_.weights_quantized = quant_info_->weights_quantized;
    quant_report_.activations_quantized = quant_info_->activations_quantized;

    if (options_.verbose_optimization) {
        std::cerr << "Quantization completed:"
                  << " mode=" << quant_mode_name(config.mode)
                  << " nodes_quantized=" << quant_report_.nodes_quantized
                  << " compression=" << quant_report_.compression_ratio << "x"
                  << "\n";
    }

    // Re-validate after quantization
    auto errors = graph_->validate();
    if (!errors.empty()) {
        throw ValidationError(errors);
    }
}

QuantizationReport InferenceSession::quantization_report() const {
    return quant_report_;
}

} // namespace pyflame_rt
