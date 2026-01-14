#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "pyflame_rt/import/importer.hpp"
#include "pyflame_rt/import/onnx_importer.hpp"
#include "pyflame_rt/import/pytorch_importer.hpp"
#include "pyflame_rt/import/torchscript_importer.hpp"
#include "pyflame_rt/import/shape_inference.hpp"

namespace py = pybind11;
using namespace pyflame_rt;
using namespace pyflame_rt::import;

void bind_import(py::module_& m) {
    // Create import submodule
    auto import_mod = m.def_submodule("import_", "Model import functionality");

    // ImportOptions
    py::class_<ImportOptions>(import_mod, "ImportOptions",
        "Options for model import")

        .def(py::init<>())

        .def_readwrite("input_shapes", &ImportOptions::input_shapes,
            "Input shapes for dynamic dimensions (dict: name -> shape)")

        .def_readwrite("infer_shapes", &ImportOptions::infer_shapes,
            "Whether to run shape inference after import")

        .def_readwrite("load_initializers", &ImportOptions::load_initializers,
            "Whether to load initializers (weights) immediately")

        .def_readwrite("opset_version", &ImportOptions::opset_version,
            "Opset version override (0 = use model's opset)")

        .def_readwrite("allow_unsupported", &ImportOptions::allow_unsupported,
            "Whether to allow unsupported operators (creates placeholder nodes)");

    // ImportStats
    py::class_<ImportStats>(import_mod, "ImportStats",
        "Statistics from import operation")

        .def(py::init<>())

        .def_readonly("total_ops", &ImportStats::total_ops,
            "Total number of operators in model")

        .def_readonly("mapped_ops", &ImportStats::mapped_ops,
            "Number of successfully mapped operators")

        .def_readonly("unsupported_ops", &ImportStats::unsupported_ops,
            "Number of unsupported operators")

        .def_readonly("unsupported_op_types", &ImportStats::unsupported_op_types,
            "List of unsupported operator types")

        .def_readonly("total_initializers", &ImportStats::total_initializers,
            "Number of initializers (weights) loaded")

        .def_readonly("total_weight_bytes", &ImportStats::total_weight_bytes,
            "Total bytes of weight data");

    // ImportResult
    py::class_<ImportResult>(import_mod, "ImportResult",
        "Result of model import")

        .def(py::init<>())

        .def("success", &ImportResult::success,
            "Check if import succeeded")

        .def_readonly("metadata", &ImportResult::metadata,
            "Model metadata")

        .def_readonly("stats", &ImportResult::stats,
            "Import statistics")

        .def_readonly("warnings", &ImportResult::warnings,
            "Warnings encountered during import")

        .def_property_readonly("graph", [](ImportResult& self) {
            return self.graph.get();
        }, py::return_value_policy::reference,
            "The imported graph (None if failed)");

    // ONNXImporter
    py::class_<ONNXImporter>(import_mod, "ONNXImporter",
        "ONNX model importer\n\n"
        "Imports ONNX models (.onnx files) into PyFlameRT format.\n"
        "Supports ONNX opsets 9-21.")

        .def(py::init<>())

        .def("import_file", &ONNXImporter::import_file,
            py::arg("path"),
            py::arg("options") = ImportOptions{},
            "Import ONNX model from file\n\n"
            "Args:\n"
            "    path: Path to .onnx file\n"
            "    options: Import options\n\n"
            "Returns:\n"
            "    ImportResult with graph and metadata")

        .def_static("get_opset_version", &ONNXImporter::get_opset_version,
            py::arg("path"),
            "Get ONNX opset version from a model file\n\n"
            "Args:\n"
            "    path: Path to ONNX model\n\n"
            "Returns:\n"
            "    Opset version, or -1 on error")

        .def_static("is_opset_supported", &ONNXImporter::is_opset_supported,
            py::arg("version"),
            "Check if opset version is supported")

        .def_property_readonly_static("MIN_OPSET_VERSION",
            [](py::object) { return ONNXImporter::MIN_OPSET_VERSION; },
            "Minimum supported ONNX opset version")

        .def_property_readonly_static("MAX_OPSET_VERSION",
            [](py::object) { return ONNXImporter::MAX_OPSET_VERSION; },
            "Maximum supported ONNX opset version");

    // PyTorchImporter
    py::class_<PyTorchImporter>(import_mod, "PyTorchImporter",
        "PyTorch checkpoint importer\n\n"
        "Imports PyTorch .pt/.pth checkpoint files.\n"
        "Requires a model definer to create the graph structure.")

        .def(py::init<>())

        .def("import_file", &PyTorchImporter::import_file,
            py::arg("path"),
            py::arg("options") = ImportOptions{},
            "Import PyTorch checkpoint from file\n\n"
            "Args:\n"
            "    path: Path to .pt/.pth file\n"
            "    options: Import options\n\n"
            "Returns:\n"
            "    ImportResult with weights loaded")

        .def("set_model_definer", &PyTorchImporter::set_model_definer,
            py::arg("definer"),
            "Set the model definition callback\n\n"
            "Args:\n"
            "    definer: Function that creates the graph architecture")

        .def("has_model_definer", &PyTorchImporter::has_model_definer,
            "Check if a model definer has been set")

        .def("clear_model_definer", &PyTorchImporter::clear_model_definer,
            "Clear the model definer")

        .def_static("is_pytorch_file", &PyTorchImporter::is_pytorch_file,
            py::arg("path"),
            "Check if file is a PyTorch checkpoint");

    // TorchScriptImporter
    py::class_<TorchScriptImporter>(import_mod, "TorchScriptImporter",
        "TorchScript model importer\n\n"
        "Imports TorchScript models (.pt, .pts files containing JIT traces/scripts).\n"
        "These are self-contained and include both architecture and weights.")

        .def(py::init<>())

        .def("import_file", &TorchScriptImporter::import_file,
            py::arg("path"),
            py::arg("options") = ImportOptions{},
            "Import TorchScript model from file\n\n"
            "Args:\n"
            "    path: Path to .pt/.pts file\n"
            "    options: Import options\n\n"
            "Returns:\n"
            "    ImportResult with graph and weights")

        .def_static("is_torchscript_file", &TorchScriptImporter::is_torchscript_file,
            py::arg("path"),
            "Check if file is a TorchScript model");

    // Convenience functions
    import_mod.def("import_model", &import_model,
        py::arg("path"),
        py::arg("options") = ImportOptions{},
        "Import a model using auto-detected format\n\n"
        "Args:\n"
        "    path: Path to model file\n"
        "    options: Import options\n\n"
        "Returns:\n"
        "    ImportResult with imported graph\n\n"
        "Raises:\n"
        "    UnsupportedFormatError if format not recognized");

    import_mod.def("supported_formats", []() {
        return ImporterRegistry::instance().supported_extensions();
    }, "Get list of supported file extensions");

    // Add from_onnx and from_pytorch convenience methods to InferenceSession
    // These are added in the main session binding
}

// Also expose at top level for convenience
void bind_import_convenience(py::module_& m) {
    // InferenceSession.from_onnx()
    m.def("from_onnx", [](const std::string& path,
                          std::optional<std::unordered_map<std::string, std::vector<int64_t>>> input_shapes,
                          std::optional<SessionOptions> session_options) {
        import::ImportOptions import_opts;
        if (input_shapes) {
            import_opts.input_shapes = input_shapes.value();
        }

        import::ONNXImporter importer;
        auto result = importer.import_file(path, import_opts);

        if (!result.success()) {
            throw std::runtime_error("Failed to import ONNX model");
        }

        // Create session from imported graph
        return std::make_unique<InferenceSession>(
            std::move(result.graph),
            session_options.value_or(SessionOptions{})
        );
    },
        py::arg("path"),
        py::arg("input_shapes") = py::none(),
        py::arg("session_options") = py::none(),
        "Create an InferenceSession from an ONNX model\n\n"
        "Args:\n"
        "    path: Path to ONNX model file\n"
        "    input_shapes: Optional dict mapping input names to shapes\n"
        "    session_options: Session configuration options\n\n"
        "Returns:\n"
        "    InferenceSession ready for inference");
}
