#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "pyflame_rt/types.hpp"
#include "pyflame_rt/options.hpp"
#include "pyflame_rt/errors.hpp"

namespace py = pybind11;
using namespace pyflame_rt;

void bind_types(py::module_& m) {
    // DType enum
    py::enum_<DType>(m, "DType", "Tensor data types")
        .value("Float32", DType::Float32, "32-bit floating point")
        .value("Float16", DType::Float16, "16-bit floating point")
        .value("BFloat16", DType::BFloat16, "Brain floating point 16")
        .value("Float64", DType::Float64, "64-bit floating point")
        .value("Int64", DType::Int64, "64-bit signed integer")
        .value("Int32", DType::Int32, "32-bit signed integer")
        .value("Int16", DType::Int16, "16-bit signed integer")
        .value("Int8", DType::Int8, "8-bit signed integer")
        .value("UInt8", DType::UInt8, "8-bit unsigned integer")
        .value("Bool", DType::Bool, "Boolean")
        .export_values();

    // OptLevel enum
    py::enum_<OptLevel>(m, "OptLevel", "Optimization levels")
        .value("NONE", OptLevel::None, "No optimization")
        .value("BASIC", OptLevel::Basic, "Constant folding, dead code elimination")
        .value("EXTENDED", OptLevel::Extended, "Operator fusion")
        .value("ALL", OptLevel::All, "All optimizations")
        .export_values();

    // TensorInfo
    py::class_<TensorInfo>(m, "TensorInfo", "Tensor metadata")
        .def(py::init<>())
        .def(py::init<std::string, Shape, DType>(),
             py::arg("name"), py::arg("shape"), py::arg("dtype"))
        .def_readwrite("name", &TensorInfo::name)
        .def_readwrite("shape", &TensorInfo::shape)
        .def_readwrite("dtype", &TensorInfo::dtype)
        .def("is_dynamic", &TensorInfo::is_dynamic, "Check if tensor has dynamic dimensions")
        .def("num_elements", &TensorInfo::num_elements, "Total number of elements")
        .def("size_bytes", &TensorInfo::size_bytes, "Size in bytes")
        .def("__repr__", [](const TensorInfo& info) {
            return "TensorInfo(name='" + info.name + "', shape=" +
                   shape_to_string(info.shape) + ", dtype=" +
                   dtype_name(info.dtype) + ")";
        });

    // NodeArg
    py::class_<NodeArg>(m, "NodeArg", "Input/output argument descriptor")
        .def_readonly("name", &NodeArg::name)
        .def_readonly("shape", &NodeArg::shape)
        .def_property_readonly("type", [](const NodeArg& arg) { return arg.type_str; })
        .def("__repr__", [](const NodeArg& arg) {
            std::string shape_str = "[";
            for (size_t i = 0; i < arg.shape.size(); ++i) {
                if (i > 0) shape_str += ", ";
                if (arg.shape[i].has_value()) {
                    shape_str += std::to_string(arg.shape[i].value());
                } else {
                    shape_str += "?";
                }
            }
            shape_str += "]";
            return "NodeArg(name='" + arg.name + "', shape=" + shape_str +
                   ", type='" + arg.type_str + "')";
        });

    // ModelMetadata
    py::class_<ModelMetadata>(m, "ModelMetadata", "Model metadata")
        .def(py::init<>())
        .def_readwrite("producer_name", &ModelMetadata::producer_name)
        .def_readwrite("producer_version", &ModelMetadata::producer_version)
        .def_readwrite("domain", &ModelMetadata::domain)
        .def_readwrite("description", &ModelMetadata::description)
        .def_readwrite("graph_name", &ModelMetadata::graph_name)
        .def_readwrite("version", &ModelMetadata::version)
        .def_readwrite("custom_metadata", &ModelMetadata::custom_metadata);

    // SessionOptions
    py::class_<SessionOptions>(m, "SessionOptions", "Session configuration options")
        .def(py::init<>())
        .def_readwrite("device", &SessionOptions::device,
                       "Device to run on: 'cpu', 'wse', 'wse2', 'wse3'")
        .def_readwrite("num_threads", &SessionOptions::num_threads,
                       "Number of CPU threads (0 = auto)")
        .def_readwrite("enable_profiling", &SessionOptions::enable_profiling,
                       "Enable performance profiling")
        .def_readwrite("execution_mode", &SessionOptions::execution_mode,
                       "Execution mode: 'sequential' or 'parallel'")
        .def_readwrite("optimization_level", &SessionOptions::optimization_level,
                       "Optimization level (OptLevel enum)")
        .def_readwrite("verbose_optimization", &SessionOptions::verbose_optimization,
                       "Enable verbose optimization logging")
        .def_readwrite("log_level", &SessionOptions::log_level,
                       "Log level: 'debug', 'info', 'warning', 'error'")
        .def("validate", &SessionOptions::validate, "Validate options");

    // RunOptions
    py::class_<RunOptions>(m, "RunOptions", "Per-run configuration options")
        .def(py::init<>())
        .def_readwrite("log_level", &RunOptions::log_level)
        .def_readwrite("tag", &RunOptions::tag)
        .def_readwrite("timeout_ms", &RunOptions::timeout_ms);

    // CompileOptions
    py::class_<CompileOptions>(m, "CompileOptions", "Model compilation options")
        .def(py::init<>())
        .def_readwrite("cache_dir", &CompileOptions::cache_dir)
        .def_readwrite("input_shapes", &CompileOptions::input_shapes)
        .def_readwrite("dynamic_batch", &CompileOptions::dynamic_batch)
        .def_readwrite("optimization_level", &CompileOptions::optimization_level);

    // Exceptions
    py::register_exception<PyFlameRTError>(m, "PyFlameRTError",
        PyExc_RuntimeError);
    py::register_exception<InvalidModelError>(m, "InvalidModelError",
        m.attr("PyFlameRTError").ptr());
    py::register_exception<UnsupportedFormatError>(m, "UnsupportedFormatError",
        m.attr("PyFlameRTError").ptr());
    py::register_exception<UnsupportedOperatorError>(m, "UnsupportedOperatorError",
        m.attr("PyFlameRTError").ptr());
    py::register_exception<ShapeMismatchError>(m, "ShapeMismatchError",
        m.attr("PyFlameRTError").ptr());
    py::register_exception<DTypeMismatchError>(m, "DTypeMismatchError",
        m.attr("PyFlameRTError").ptr());
    py::register_exception<ValidationError>(m, "ValidationError",
        m.attr("PyFlameRTError").ptr());
    py::register_exception<BackendError>(m, "BackendError",
        m.attr("PyFlameRTError").ptr());
    py::register_exception<InputError>(m, "InputError",
        m.attr("PyFlameRTError").ptr());
}
