#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "pyflame_rt/session.hpp"

namespace py = pybind11;
using namespace pyflame_rt;

// Forward declarations from tensor_bind.cpp
Tensor tensor_from_numpy(py::array arr);
py::array tensor_to_numpy(const Tensor& tensor);

void bind_session(py::module_& m) {
    py::class_<InferenceSession>(m, "InferenceSession",
        "High-level inference session for model execution.\n"
        "Provides an ONNX Runtime-compatible API.")

        .def(py::init([](const std::string& model_path,
                        std::optional<SessionOptions> options,
                        std::optional<std::vector<std::string>> providers) {
            return std::make_unique<InferenceSession>(
                model_path,
                options.value_or(SessionOptions{}),
                providers.value_or(std::vector<std::string>{}));
        }),
            py::arg("model_path"),
            py::arg("options") = py::none(),
            py::arg("providers") = py::none(),
            "Load a model and prepare for inference.\n\n"
            "Args:\n"
            "    model_path: Path to model file (.pfm)\n"
            "    options: Session configuration options\n"
            "    providers: Execution provider priority list")

        .def("run", [](InferenceSession& self,
                       std::optional<std::vector<std::string>> output_names,
                       const py::dict& input_feed,
                       std::optional<RunOptions> run_options) {
            // Convert input dict to map of Tensors (with GIL held)
            std::unordered_map<std::string, Tensor> feeds;
            for (auto item : input_feed) {
                std::string name = py::str(item.first);

                // Security fix CRIT-B1: Validate input type before conversion
                if (!py::isinstance<py::array>(item.second) &&
                    !py::isinstance<py::buffer>(item.second)) {
                    // Try to convert to array
                    py::array arr = py::array::ensure(item.second);
                    if (!arr) {
                        throw std::runtime_error(
                            "Input '" + name + "' could not be converted to numpy array. "
                            "Expected array-like object.");
                    }
                    feeds[name] = tensor_from_numpy(arr);
                } else {
                    py::array arr = py::array::ensure(item.second);
                    if (!arr) {
                        throw std::runtime_error(
                            "Input '" + name + "' array conversion failed");
                    }
                    feeds[name] = tensor_from_numpy(arr);
                }
            }

            std::vector<Tensor> outputs;
            {
                // Security fix HIGH-C4: Release GIL during inference
                py::gil_scoped_release release;
                outputs = self.run(
                    output_names.value_or(std::vector<std::string>{}),
                    feeds,
                    run_options.value_or(RunOptions{})
                );
            }

            // Convert outputs to numpy arrays (with GIL re-acquired)
            py::list result;
            for (const auto& tensor : outputs) {
                result.append(tensor_to_numpy(tensor));
            }
            return result;
        },
            py::arg("output_names") = py::none(),
            py::arg("input_feed"),
            py::arg("run_options") = py::none(),
            "Run inference on the model.\n\n"
            "Args:\n"
            "    output_names: Names of outputs to return (None = all outputs)\n"
            "    input_feed: Dictionary mapping input names to numpy arrays\n"
            "    run_options: Per-run configuration\n\n"
            "Returns:\n"
            "    List of output numpy arrays")

        .def("get_inputs", &InferenceSession::get_inputs,
             "Get input tensor metadata")

        .def("get_outputs", &InferenceSession::get_outputs,
             "Get output tensor metadata")

        .def("get_modelmeta", &InferenceSession::get_modelmeta,
             "Get model metadata")

        .def("get_providers", &InferenceSession::get_providers,
             "Get list of available execution providers")

        .def_property_readonly("_graph_name", [](const InferenceSession& self) {
            return self.graph().name();
        }, "Internal: graph name");

    // Convenience function matching ONNX Runtime style
    m.def("create_session", [](const std::string& model_path,
                               std::optional<SessionOptions> options,
                               std::optional<std::vector<std::string>> providers) {
        return std::make_unique<InferenceSession>(
            model_path,
            options.value_or(SessionOptions{}),
            providers.value_or(std::vector<std::string>{})
        );
    },
        py::arg("model_path"),
        py::arg("options") = py::none(),
        py::arg("providers") = py::none(),
        "Create an inference session (alias for InferenceSession constructor)");
}
