#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include "pyflame_rt/custom/custom_op.hpp"

namespace py = pybind11;
using namespace pyflame_rt;
using namespace pyflame_rt::custom;

void bind_custom_ops(py::module_& m) {
    auto custom = m.def_submodule("custom",
        "Custom operator registration and execution");

    // ========================================================================
    // Enums
    // ========================================================================

    py::enum_<BackendType>(custom, "BackendType",
        "Target backend for custom operators")
        .value("CPU", BackendType::CPU, "CPU execution")
        .value("WSE", BackendType::WSE, "Cerebras WSE")
        .value("CUDA", BackendType::CUDA, "NVIDIA CUDA")
        .value("ALL", BackendType::All, "All backends");

    // ========================================================================
    // OpInput / OpOutput
    // ========================================================================

    py::class_<OpInput>(custom, "OpInput",
        "Input specification for custom operator")
        .def(py::init<>())
        .def_readwrite("name", &OpInput::name, "Input name")
        .def_readwrite("dtype", &OpInput::dtype, "Data type")
        .def_readwrite("optional", &OpInput::optional, "Is optional")
        .def_readwrite("variadic", &OpInput::variadic, "Is variadic");

    py::class_<OpOutput>(custom, "OpOutput",
        "Output specification for custom operator")
        .def(py::init<>())
        .def_readwrite("name", &OpOutput::name, "Output name")
        .def_readwrite("dtype", &OpOutput::dtype, "Data type");

    // ========================================================================
    // OpAttribute
    // ========================================================================

    py::class_<OpAttribute>(custom, "OpAttribute",
        "Attribute specification for custom operator")
        .def(py::init<>())
        .def_readwrite("name", &OpAttribute::name, "Attribute name")
        .def_readwrite("type", &OpAttribute::type, "Attribute type")
        .def_readwrite("required", &OpAttribute::required, "Is required")
        .def_readwrite("description", &OpAttribute::description, "Description");

    // ========================================================================
    // OpSchema
    // ========================================================================

    py::class_<OpSchema>(custom, "OpSchema",
        "Schema defining a custom operator")
        .def(py::init<>())
        .def_readwrite("name", &OpSchema::name, "Operator name")
        .def_readwrite("domain", &OpSchema::domain, "Operator domain")
        .def_readwrite("version", &OpSchema::version, "Operator version")
        .def_readwrite("doc", &OpSchema::doc, "Documentation string")
        .def_readwrite("inputs", &OpSchema::inputs, "Input specifications")
        .def_readwrite("outputs", &OpSchema::outputs, "Output specifications")
        .def_readwrite("attributes", &OpSchema::attributes, "Attribute specifications");

    // ========================================================================
    // CustomOp
    // ========================================================================

    py::class_<CustomOp>(custom, "CustomOp",
        "Custom operator instance")
        .def("name", &CustomOp::name, "Get operator name")
        .def("domain", &CustomOp::domain, "Get operator domain")
        .def("full_name", &CustomOp::full_name, "Get fully qualified name")
        .def_property_readonly("schema", &CustomOp::schema, "Get operator schema")
        .def("execute", &CustomOp::execute,
            py::arg("inputs"), py::arg("attributes"),
            "Execute the operator")
        .def("infer_output_shapes", &CustomOp::infer_output_shapes,
            py::arg("input_shapes"),
            "Infer output shapes from input shapes")
        .def("infer_output_dtypes", &CustomOp::infer_output_dtypes,
            py::arg("input_dtypes"),
            "Infer output dtypes from input dtypes")
        .def("has_gradient", &CustomOp::has_gradient,
            "Check if gradient function is registered")
        .def("gradient", &CustomOp::gradient,
            py::arg("inputs"), py::arg("grad_outputs"),
            "Compute gradients")
        .def("supports_backend", &CustomOp::supports_backend,
            py::arg("backend"),
            "Check if backend is supported")
        .def("get_supported_backends", &CustomOp::get_supported_backends,
            "Get list of supported backends");

    // ========================================================================
    // CustomOpRegistry
    // ========================================================================

    py::class_<CustomOpRegistry>(custom, "CustomOpRegistry",
        "Registry for custom operators")
        .def_static("instance", &CustomOpRegistry::instance,
            py::return_value_policy::reference,
            "Get singleton instance")
        .def("register_op", &CustomOpRegistry::register_op,
            py::arg("schema"),
            py::return_value_policy::reference,
            "Register a new custom operator")
        .def("get",
            py::overload_cast<const std::string&>(&CustomOpRegistry::get),
            py::arg("name"),
            py::return_value_policy::reference,
            "Get operator by name")
        .def("get_by_domain",
            py::overload_cast<const std::string&, const std::string&>(&CustomOpRegistry::get),
            py::arg("domain"), py::arg("name"),
            py::return_value_policy::reference,
            "Get operator by domain and name")
        .def("has",
            py::overload_cast<const std::string&>(&CustomOpRegistry::has, py::const_),
            py::arg("name"),
            "Check if operator exists")
        .def("list",
            py::overload_cast<>(&CustomOpRegistry::list, py::const_),
            "List all registered operators")
        .def("list_domain",
            py::overload_cast<const std::string&>(&CustomOpRegistry::list, py::const_),
            py::arg("domain"),
            "List operators in domain")
        .def("unregister",
            py::overload_cast<const std::string&>(&CustomOpRegistry::unregister),
            py::arg("name"),
            "Unregister an operator")
        .def("clear", &CustomOpRegistry::clear,
            "Clear all registered operators")
        .def("size", &CustomOpRegistry::size,
            "Get number of registered operators");

    // ========================================================================
    // CustomOpBuilder
    // ========================================================================

    py::class_<CustomOpBuilder>(custom, "CustomOpBuilder",
        "Fluent builder for custom operators")
        .def(py::init<const std::string&>(),
            py::arg("name"),
            "Create builder with operator name")
        .def("domain", &CustomOpBuilder::domain,
            py::arg("domain"),
            "Set operator domain")
        .def("version", &CustomOpBuilder::version,
            py::arg("version"),
            "Set operator version")
        .def("doc", &CustomOpBuilder::doc,
            py::arg("documentation"),
            "Set documentation string")
        .def("input", &CustomOpBuilder::input,
            py::arg("name"), py::arg("dtype") = DType::Float32,
            py::arg("optional") = false,
            "Add input specification")
        .def("variadic_input", &CustomOpBuilder::variadic_input,
            py::arg("name"), py::arg("dtype") = DType::Float32,
            "Add variadic input specification")
        .def("output", &CustomOpBuilder::output,
            py::arg("name"), py::arg("dtype") = DType::Float32,
            "Add output specification")
        .def("attr_int", &CustomOpBuilder::attr_int,
            py::arg("name"), py::arg("required") = false,
            "Add integer attribute")
        .def("attr_float", &CustomOpBuilder::attr_float,
            py::arg("name"), py::arg("required") = false,
            "Add float attribute")
        .def("attr_string", &CustomOpBuilder::attr_string,
            py::arg("name"), py::arg("required") = false,
            "Add string attribute")
        .def("attr_ints", &CustomOpBuilder::attr_ints,
            py::arg("name"), py::arg("required") = false,
            "Add integer array attribute")
        .def("attr_floats", &CustomOpBuilder::attr_floats,
            py::arg("name"), py::arg("required") = false,
            "Add float array attribute")
        .def("kernel", &CustomOpBuilder::kernel,
            py::arg("kernel_fn"), py::arg("backend") = BackendType::All,
            "Set kernel function")
        .def("shape_inference", &CustomOpBuilder::shape_inference,
            py::arg("inference_fn"),
            "Set shape inference function")
        .def("type_inference", &CustomOpBuilder::type_inference,
            py::arg("inference_fn"),
            "Set type inference function")
        .def("gradient", &CustomOpBuilder::gradient,
            py::arg("gradient_fn"),
            "Set gradient function")
        .def("build", &CustomOpBuilder::build,
            py::return_value_policy::reference,
            "Build and register the operator");

    // ========================================================================
    // Utility Functions
    // ========================================================================

    custom.def("register_custom_op", [](
        const std::string& name,
        const std::string& domain,
        const std::vector<std::string>& input_names,
        const std::vector<std::string>& output_names,
        py::function kernel_fn) {

        OpSchema schema;
        schema.name = name;
        schema.domain = domain;

        for (const auto& in_name : input_names) {
            OpInput input;
            input.name = in_name;
            input.dtype = DType::Float32;
            schema.inputs.push_back(input);
        }

        for (const auto& out_name : output_names) {
            OpOutput output;
            output.name = out_name;
            output.dtype = DType::Float32;
            schema.outputs.push_back(output);
        }

        CustomOp& op = CustomOpRegistry::instance().register_op(schema);

        // Wrap Python function
        // Security: copy kernel_fn to ensure lifetime (HIGH-C3 fix)
        py::function kernel_fn_copy = kernel_fn;
        op.set_kernel([kernel_fn_copy](
            const std::vector<Tensor>& inputs,
            const std::unordered_map<std::string, std::any>& attrs) -> std::vector<Tensor> {
            py::gil_scoped_acquire acquire;

            // Security: wrap in try-catch to handle Python exceptions (HIGH-C4 fix)
            try {
                py::list py_inputs;
                for (const auto& t : inputs) {
                    py_inputs.append(t);
                }
                py::dict py_attrs;
                // Note: attrs conversion would need more work for full support
                py::object result = kernel_fn_copy(py_inputs, py_attrs);

                // Security: validate result (MED-C2 fix)
                if (result.is_none()) {
                    throw std::runtime_error("Custom op kernel returned None");
                }

                std::vector<Tensor> outputs;
                if (py::isinstance<py::list>(result)) {
                    for (auto item : result.cast<py::list>()) {
                        if (!py::isinstance<Tensor>(item)) {
                            throw std::runtime_error("Custom op kernel list must contain Tensors");
                        }
                        outputs.push_back(item.cast<Tensor>());
                    }
                } else if (py::isinstance<Tensor>(result)) {
                    outputs.push_back(result.cast<Tensor>());
                } else {
                    throw std::runtime_error("Custom op kernel must return Tensor or list of Tensors");
                }
                return outputs;
            } catch (const py::error_already_set& e) {
                // Re-throw Python exceptions as C++ exceptions
                throw std::runtime_error(std::string("Python exception in custom op: ") + e.what());
            }
        }, BackendType::CPU);

        return &op;
    },
    py::arg("name"),
    py::arg("domain") = "custom",
    py::arg("input_names") = std::vector<std::string>{"input"},
    py::arg("output_names") = std::vector<std::string>{"output"},
    py::arg("kernel_fn"),
    py::return_value_policy::reference,
    "Register a custom operator from Python");

    custom.def("get_op", [](const std::string& name) {
        return CustomOpRegistry::instance().get(name);
    },
    py::arg("name"),
    py::return_value_policy::reference,
    "Get a custom operator by name");

    custom.def("list_ops", []() {
        return CustomOpRegistry::instance().list();
    },
    "List all custom operators");
}
