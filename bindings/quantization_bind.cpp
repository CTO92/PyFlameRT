#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include "pyflame_rt/quantization/quant_config.hpp"
#include "pyflame_rt/quantization/quant_params.hpp"
#include "pyflame_rt/quantization/calibrator.hpp"
#include "pyflame_rt/quantization/quantizer.hpp"
#include "pyflame_rt/quantization/quant_ops.hpp"

namespace py = pybind11;
using namespace pyflame_rt;
using namespace pyflame_rt::quantization;

void bind_quantization(py::module_& m) {
    auto quant = m.def_submodule("quantization",
        "Quantization support for reduced-precision inference");

    // ========================================================================
    // Enums
    // ========================================================================

    py::enum_<QuantMode>(quant, "QuantMode",
        "Quantization mode for model weights and activations")
        .value("NONE", QuantMode::None, "No quantization (FP32)")
        .value("FP16", QuantMode::FP16, "Float16 precision")
        .value("BFLOAT16", QuantMode::BFloat16, "BFloat16 precision")
        .value("DYNAMIC_INT8", QuantMode::DynamicInt8,
            "Dynamic INT8 quantization")
        .value("STATIC_INT8", QuantMode::StaticInt8,
            "Static INT8 with calibration");

    py::enum_<QuantGranularity>(quant, "QuantGranularity",
        "Granularity of quantization parameters")
        .value("PER_TENSOR", QuantGranularity::PerTensor,
            "Single scale/zero-point per tensor")
        .value("PER_CHANNEL", QuantGranularity::PerChannel,
            "Scale/zero-point per output channel");

    py::enum_<CalibrationMethod>(quant, "CalibrationMethod",
        "Algorithm for computing quantization parameters")
        .value("MINMAX", CalibrationMethod::MinMax,
            "Use min/max of observed values")
        .value("ENTROPY", CalibrationMethod::Entropy,
            "KL-divergence based calibration")
        .value("PERCENTILE", CalibrationMethod::Percentile,
            "Use percentile bounds");

    // ========================================================================
    // QuantConfig
    // ========================================================================

    py::class_<QuantConfig>(quant, "QuantConfig",
        "Configuration for quantization")
        .def(py::init<>())
        .def_readwrite("mode", &QuantConfig::mode,
            "Quantization mode")
        .def_readwrite("weight_dtype", &QuantConfig::weight_dtype,
            "Data type for quantized weights")
        .def_readwrite("activation_dtype", &QuantConfig::activation_dtype,
            "Data type for quantized activations")
        .def_readwrite("granularity", &QuantConfig::granularity,
            "Quantization granularity")
        .def_readwrite("calibration_method", &QuantConfig::calibration_method,
            "Calibration algorithm")
        .def_readwrite("calibration_samples", &QuantConfig::calibration_samples,
            "Number of calibration samples")
        .def_readwrite("calibration_percentile", &QuantConfig::calibration_percentile,
            "Percentile for percentile calibration")
        .def_readwrite("quantize_inputs", &QuantConfig::quantize_inputs,
            "Whether to quantize input tensors")
        .def_readwrite("quantize_outputs", &QuantConfig::quantize_outputs,
            "Whether to quantize output tensors")
        .def_readwrite("symmetric", &QuantConfig::symmetric,
            "Use symmetric quantization")
        .def("add_exclude_op", [](QuantConfig& self, const std::string& op) {
            self.exclude_ops.insert(op);
        }, "Add operator type to exclude from quantization")
        .def("add_exclude_node", [](QuantConfig& self, const std::string& node) {
            self.exclude_nodes.insert(node);
        }, "Add node name to exclude from quantization")
        .def("is_valid", &QuantConfig::is_valid,
            "Check if configuration is valid")
        .def("validation_error", &QuantConfig::validation_error,
            "Get validation error message")
        .def("is_int8_mode", &QuantConfig::is_int8_mode,
            "Check if this is an INT8 quantization mode")
        .def("is_float_mode", &QuantConfig::is_float_mode,
            "Check if this is a float reduction mode (FP16/BF16)")
        .def("requires_calibration", &QuantConfig::requires_calibration,
            "Check if calibration is required")
        .def_static("fp16", &QuantConfig::fp16,
            "Create FP16 quantization config")
        .def_static("bfloat16", &QuantConfig::bfloat16,
            "Create BFloat16 quantization config")
        .def_static("dynamic_int8", &QuantConfig::dynamic_int8,
            "Create dynamic INT8 quantization config")
        .def_static("static_int8", &QuantConfig::static_int8,
            py::arg("calibration_samples") = 100,
            "Create static INT8 quantization config");

    // ========================================================================
    // QuantParams
    // ========================================================================

    py::class_<QuantParams>(quant, "QuantParams",
        "Quantization parameters (scale and zero-point)")
        .def(py::init<>())
        .def_readonly("scales", &QuantParams::scales,
            "Scale factor(s)")
        .def_readonly("zero_points", &QuantParams::zero_points,
            "Zero point(s)")
        .def_readonly("quantized_dtype", &QuantParams::quantized_dtype,
            "Quantized data type")
        .def_readonly("channel_axis", &QuantParams::channel_axis,
            "Channel axis for per-channel quantization (-1 for per-tensor)")
        .def_readonly("symmetric", &QuantParams::symmetric,
            "Whether symmetric quantization is used")
        .def("is_per_tensor", &QuantParams::is_per_tensor,
            "Check if per-tensor quantization")
        .def("is_per_channel", &QuantParams::is_per_channel,
            "Check if per-channel quantization")
        .def("num_params", &QuantParams::num_params,
            "Get number of scale/zero-point pairs")
        .def("is_valid", &QuantParams::is_valid,
            "Check if parameters are valid")
        .def_static("per_tensor", &QuantParams::per_tensor,
            py::arg("scale"), py::arg("zero_point"),
            py::arg("dtype") = DType::Int8,
            "Create per-tensor quantization params")
        .def_static("per_channel", &QuantParams::per_channel,
            py::arg("scales"), py::arg("zero_points"),
            py::arg("axis"), py::arg("dtype") = DType::Int8,
            "Create per-channel quantization params")
        .def_static("compute_from_minmax", &QuantParams::compute_from_minmax,
            py::arg("min_val"), py::arg("max_val"),
            py::arg("dtype"), py::arg("symmetric"),
            "Compute params from min/max values");

    // ========================================================================
    // GraphQuantInfo
    // ========================================================================

    py::class_<GraphQuantInfo>(quant, "GraphQuantInfo",
        "Quantization info for all tensors in a graph")
        .def(py::init<>())
        .def_readwrite("mode", &GraphQuantInfo::mode,
            "Quantization mode")
        .def_readwrite("weights_quantized", &GraphQuantInfo::weights_quantized,
            "Whether weights are quantized")
        .def_readwrite("activations_quantized", &GraphQuantInfo::activations_quantized,
            "Whether activations are quantized")
        .def("get_params", &GraphQuantInfo::get_params,
            py::arg("name"), py::return_value_policy::reference,
            "Get params for a tensor")
        .def("set_params", &GraphQuantInfo::set_params,
            py::arg("name"), py::arg("params"),
            "Set params for a tensor")
        .def("has_params", &GraphQuantInfo::has_params,
            py::arg("name"),
            "Check if tensor has quantization params")
        .def("num_quantized", &GraphQuantInfo::num_quantized,
            "Get number of quantized tensors")
        .def("clear", &GraphQuantInfo::clear,
            "Clear all quantization params");

    // ========================================================================
    // QuantizationResult::Stats
    // ========================================================================

    py::class_<QuantizationResult::Stats>(quant, "QuantizationStats",
        "Statistics from quantization")
        .def_readonly("nodes_quantized",
            &QuantizationResult::Stats::nodes_quantized,
            "Number of nodes quantized")
        .def_readonly("nodes_skipped",
            &QuantizationResult::Stats::nodes_skipped,
            "Number of nodes skipped")
        .def_readonly("weights_quantized",
            &QuantizationResult::Stats::weights_quantized,
            "Number of weights quantized")
        .def_readonly("original_size_bytes",
            &QuantizationResult::Stats::original_size_bytes,
            "Original size in bytes")
        .def_readonly("quantized_size_bytes",
            &QuantizationResult::Stats::quantized_size_bytes,
            "Quantized size in bytes")
        .def("compression_ratio",
            &QuantizationResult::Stats::compression_ratio,
            "Get compression ratio")
        .def("original_size_mb",
            &QuantizationResult::Stats::original_size_mb,
            "Get original size in MB")
        .def("quantized_size_mb",
            &QuantizationResult::Stats::quantized_size_mb,
            "Get quantized size in MB");

    // ========================================================================
    // QuantizationResult
    // ========================================================================

    py::class_<QuantizationResult>(quant, "QuantizationResult",
        "Result of quantization transformation")
        .def_readonly("stats", &QuantizationResult::stats,
            "Quantization statistics")
        .def_readonly("success", &QuantizationResult::success,
            "Whether quantization succeeded")
        .def_readonly("error_message", &QuantizationResult::error_message,
            "Error message if quantization failed")
        .def_property_readonly("quant_info",
            [](const QuantizationResult& r) { return r.quant_info; },
            "Quantization info for all tensors");

    // ========================================================================
    // Quantizer
    // ========================================================================

    py::class_<Quantizer>(quant, "Quantizer",
        "Graph quantizer for transforming models to quantized versions")
        .def(py::init<const QuantConfig&>(),
            py::arg("config"),
            "Create quantizer with configuration")
        .def("supports_quantization", &Quantizer::supports_quantization,
            py::arg("op_type"),
            "Check if operator supports quantization")
        .def("get_quantizable_ops", &Quantizer::get_quantizable_ops,
            "Get list of quantizable operators")
        .def("is_excluded", &Quantizer::is_excluded,
            py::arg("op_type"), py::arg("node_name") = "",
            "Check if operator/node is excluded")
        .def("convert_to_fp16", &Quantizer::convert_to_fp16,
            py::arg("graph"),
            "Convert graph to FP16")
        .def("convert_to_bfloat16", &Quantizer::convert_to_bfloat16,
            py::arg("graph"),
            "Convert graph to BFloat16")
        .def("quantize_dynamic", &Quantizer::quantize_dynamic,
            py::arg("graph"),
            "Quantize graph with dynamic quantization")
        .def("quantize", &Quantizer::quantize,
            py::arg("graph"), py::arg("quant_info"),
            "Quantize graph with pre-computed quantization info")
        .def_property_readonly("config",
            [](const Quantizer& q) { return q.config(); },
            "Get quantizer configuration");

    // ========================================================================
    // CalibrationStats
    // ========================================================================

    py::class_<CalibrationStats>(quant, "CalibrationStats",
        "Statistics collected during calibration")
        .def(py::init<>())
        .def_readonly("min_val", &CalibrationStats::min_val,
            "Minimum observed value")
        .def_readonly("max_val", &CalibrationStats::max_val,
            "Maximum observed value")
        .def_readonly("num_samples", &CalibrationStats::num_samples,
            "Number of samples collected")
        .def("mean", &CalibrationStats::mean,
            "Get mean of observed values")
        .def("variance", &CalibrationStats::variance,
            "Get variance of observed values")
        .def("stddev", &CalibrationStats::stddev,
            "Get standard deviation")
        .def("compute_minmax_params", &CalibrationStats::compute_minmax_params,
            py::arg("dtype"), py::arg("symmetric"),
            "Compute params using min-max method")
        .def("compute_percentile_params", &CalibrationStats::compute_percentile_params,
            py::arg("dtype"), py::arg("symmetric"), py::arg("percentile"),
            "Compute params using percentile method")
        .def("compute_entropy_params", &CalibrationStats::compute_entropy_params,
            py::arg("dtype"), py::arg("symmetric"),
            "Compute params using entropy method");

    // ========================================================================
    // Calibrator
    // ========================================================================

    py::class_<Calibrator>(quant, "Calibrator",
        "Calibrator for static quantization")
        .def(py::init<const Graph&, const QuantConfig&>(),
            py::arg("graph"), py::arg("config"),
            "Create calibrator for a graph")
        .def("calibrate", &Calibrator::calibrate,
            py::arg("data_provider"), py::arg("num_batches"),
            "Run calibration with data provider")
        .def("add_sample", &Calibrator::add_sample,
            py::arg("tensor_values"),
            "Add calibration sample manually")
        .def("compute_quant_params", &Calibrator::compute_quant_params,
            "Compute final quantization parameters")
        .def("is_calibrated", &Calibrator::is_calibrated,
            "Check if calibration is complete")
        .def("num_samples", &Calibrator::num_samples,
            "Get number of samples processed")
        .def_property_readonly("quant_info",
            [](const Calibrator& c) { return c.quant_info(); },
            "Get computed quantization info")
        .def("register_tensor", &Calibrator::register_tensor,
            py::arg("name"), py::arg("shape"),
            "Register tensor for calibration")
        .def("get_registered_tensors", &Calibrator::get_registered_tensors,
            "Get list of registered tensors");

    // ========================================================================
    // Utility Functions
    // ========================================================================

    quant.def("is_int8_quantizable", &is_int8_quantizable,
        py::arg("op_type"),
        "Check if operator type supports INT8 quantization");

    quant.def("requires_float_precision", &requires_float_precision,
        py::arg("op_type"),
        "Check if operator requires float precision");

    quant.def("get_quantized_op_type", &get_quantized_op_type,
        py::arg("op_type"),
        "Get quantized version of operator type");

    quant.def("quant_mode_name", &quant_mode_name,
        py::arg("mode"),
        "Convert QuantMode to string");

    quant.def("granularity_name", &granularity_name,
        py::arg("granularity"),
        "Convert QuantGranularity to string");

    quant.def("calibration_method_name", &calibration_method_name,
        py::arg("method"),
        "Convert CalibrationMethod to string");
}
