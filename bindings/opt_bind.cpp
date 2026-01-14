#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "pyflame_rt/opt/passes.hpp"

namespace py = pybind11;
using namespace pyflame_rt;
using namespace pyflame_rt::opt;

void bind_opt(py::module_& m) {
    auto opt_module = m.def_submodule("opt", "Graph optimization passes");

    // ========================================================================
    // PassStats
    // ========================================================================
    py::class_<PassStats>(opt_module, "PassStats",
        "Statistics from an optimization pass")
        .def(py::init<>())
        .def_readwrite("nodes_removed", &PassStats::nodes_removed,
            "Number of nodes removed")
        .def_readwrite("nodes_added", &PassStats::nodes_added,
            "Number of nodes added")
        .def_readwrite("nodes_fused", &PassStats::nodes_fused,
            "Number of nodes fused")
        .def_readwrite("constants_folded", &PassStats::constants_folded,
            "Number of constants folded")
        .def_readwrite("initializers_removed", &PassStats::initializers_removed,
            "Number of initializers removed")
        .def("__repr__", [](const PassStats& s) {
            return "<PassStats removed=" + std::to_string(s.nodes_removed) +
                   " added=" + std::to_string(s.nodes_added) +
                   " fused=" + std::to_string(s.nodes_fused) +
                   " folded=" + std::to_string(s.constants_folded) + ">";
        });

    // ========================================================================
    // PassResult
    // ========================================================================
    py::class_<PassResult>(opt_module, "PassResult",
        "Result of running optimization pass(es)")
        .def(py::init<>())
        .def_readwrite("modified", &PassResult::modified,
            "Whether the graph was modified")
        .def_readwrite("stats", &PassResult::stats,
            "Statistics about changes made")
        .def_readwrite("warnings", &PassResult::warnings,
            "Warnings generated during optimization")
        .def("__repr__", [](const PassResult& r) {
            return "<PassResult modified=" + std::string(r.modified ? "True" : "False") +
                   " nodes_removed=" + std::to_string(r.stats.nodes_removed) + ">";
        });

    // ========================================================================
    // PassManagerConfig
    // ========================================================================
    py::class_<PassManagerConfig>(opt_module, "PassManagerConfig",
        "Configuration for the pass manager")
        .def(py::init<>())
        .def_readwrite("opt_level", &PassManagerConfig::opt_level,
            "Optimization level (OptLevel enum)")
        .def_readwrite("max_iterations", &PassManagerConfig::max_iterations,
            "Maximum iterations for fixed-point optimization")
        .def_readwrite("verbose", &PassManagerConfig::verbose,
            "Enable verbose logging")
        .def_readwrite("skip_passes", &PassManagerConfig::skip_passes,
            "List of pass names to skip")
        .def_readwrite("only_passes", &PassManagerConfig::only_passes,
            "Run only these passes (empty = all)")
        .def_readwrite("validate_after_pass", &PassManagerConfig::validate_after_pass,
            "Validate graph after each pass");

    // ========================================================================
    // PassManager
    // ========================================================================
    py::class_<PassManager>(opt_module, "PassManager",
        "Manages and executes optimization passes")
        .def(py::init<PassManagerConfig>(),
             py::arg("config") = PassManagerConfig{},
             "Create pass manager with configuration")
        .def("run", &PassManager::run,
             py::arg("graph"),
             "Run all applicable passes on the graph")
        .def("run_pass", &PassManager::run_pass,
             py::arg("name"), py::arg("graph"),
             "Run a specific pass by name")
        .def("run_until_fixed_point", &PassManager::run_until_fixed_point,
             py::arg("graph"),
             "Run passes repeatedly until no more changes")
        .def("list_passes", &PassManager::list_passes,
             "Get list of registered pass names")
        .def("has_pass", &PassManager::has_pass,
             py::arg("name"),
             "Check if pass is registered")
        .def_static("create_default", &PassManager::create_default,
             py::arg("level") = OptLevel::Extended,
             "Create pass manager with built-in passes")
        .def_property("config",
             &PassManager::config,
             &PassManager::set_config,
             "Pass manager configuration");

    // ========================================================================
    // Individual Pass Configs
    // ========================================================================

    py::class_<ConstantFoldingConfig>(opt_module, "ConstantFoldingConfig",
        "Configuration for constant folding pass")
        .def(py::init<>())
        .def_readwrite("max_tensor_bytes", &ConstantFoldingConfig::max_tensor_bytes,
            "Maximum tensor size to fold (bytes)")
        .def_readwrite("fold_shape_ops", &ConstantFoldingConfig::fold_shape_ops,
            "Fold shape-computing operations")
        .def_readwrite("fold_expensive_ops", &ConstantFoldingConfig::fold_expensive_ops,
            "Fold expensive operations (MatMul, Conv)")
        .def_readwrite("exclude_ops", &ConstantFoldingConfig::exclude_ops,
            "Operators to never fold");

    py::class_<DCEConfig>(opt_module, "DCEConfig",
        "Configuration for dead code elimination pass")
        .def(py::init<>())
        .def_readwrite("remove_initializers", &DCEConfig::remove_initializers,
            "Remove unused initializers")
        .def_readwrite("remove_identity", &DCEConfig::remove_identity,
            "Remove identity nodes")
        .def_readwrite("remove_dropout", &DCEConfig::remove_dropout,
            "Remove dropout nodes in inference mode");

    py::class_<CSEConfig>(opt_module, "CSEConfig",
        "Configuration for common subexpression elimination pass")
        .def(py::init<>())
        .def_readwrite("check_attributes", &CSEConfig::check_attributes,
            "Include attributes in equivalence check")
        .def_readwrite("max_comparisons", &CSEConfig::max_comparisons,
            "Maximum comparisons per node");

    py::class_<FusionConfig>(opt_module, "FusionConfig",
        "Configuration for operator fusion pass")
        .def(py::init<>())
        .def_readwrite("fuse_conv_bn", &FusionConfig::fuse_conv_bn,
            "Enable Conv + BatchNorm fusion")
        .def_readwrite("fuse_conv_bn_activation", &FusionConfig::fuse_conv_bn_activation,
            "Enable Conv + BN + Activation fusion")
        .def_readwrite("fuse_matmul_add", &FusionConfig::fuse_matmul_add,
            "Enable MatMul + Add fusion")
        .def_readwrite("fuse_elementwise_activation", &FusionConfig::fuse_elementwise_activation,
            "Enable element-wise + activation fusion")
        .def_readwrite("check_backend_support", &FusionConfig::check_backend_support,
            "Only fuse if backend supports fused op")
        .def_readwrite("target_backend", &FusionConfig::target_backend,
            "Target backend name");

    py::class_<LayoutConfig>(opt_module, "LayoutConfig",
        "Configuration for layout optimization pass")
        .def(py::init<>())
        .def_readwrite("insert_transposes", &LayoutConfig::insert_transposes,
            "Insert transpose nodes when needed")
        .def_readwrite("propagate_layout", &LayoutConfig::propagate_layout,
            "Propagate layout through graph")
        .def_readwrite("target_backend", &LayoutConfig::target_backend,
            "Target backend name");

    // ========================================================================
    // Layout enum
    // ========================================================================
    py::enum_<Layout>(opt_module, "Layout", "Tensor memory layout")
        .value("NCHW", Layout::NCHW, "Batch, Channel, Height, Width (PyTorch)")
        .value("NHWC", Layout::NHWC, "Batch, Height, Width, Channel (TensorFlow)")
        .value("NC4HW4", Layout::NC4HW4, "Blocked format for SIMD")
        .value("Unknown", Layout::Unknown, "Unknown layout");

    // ========================================================================
    // Convenience functions
    // ========================================================================

    opt_module.def("optimize",
        [](Graph& graph, OptLevel level) {
            PassManager pm(PassManagerConfig{.opt_level = level});
            // Register built-in passes manually since create_default
            // relies on the registry
            pm.register_pass(std::make_unique<ConstantFoldingPass>());
            pm.register_pass(std::make_unique<DeadCodeEliminationPass>());
            pm.register_pass(std::make_unique<CSEPass>());
            pm.register_pass(std::make_unique<OperatorFusionPass>());
            pm.register_pass(std::make_unique<LayoutOptimizationPass>());
            return pm.run_until_fixed_point(graph);
        },
        py::arg("graph"),
        py::arg("level") = OptLevel::Extended,
        "Optimize graph with default passes at specified level");

    opt_module.def("fold_constants",
        [](Graph& graph, const ConstantFoldingConfig& config) {
            ConstantFoldingPass pass(config);
            return pass.run(graph);
        },
        py::arg("graph"),
        py::arg("config") = ConstantFoldingConfig{},
        "Run constant folding on graph");

    opt_module.def("eliminate_dead_code",
        [](Graph& graph, const DCEConfig& config) {
            DeadCodeEliminationPass pass(config);
            return pass.run(graph);
        },
        py::arg("graph"),
        py::arg("config") = DCEConfig{},
        "Run dead code elimination on graph");

    opt_module.def("eliminate_common_subexpressions",
        [](Graph& graph, const CSEConfig& config) {
            CSEPass pass(config);
            return pass.run(graph);
        },
        py::arg("graph"),
        py::arg("config") = CSEConfig{},
        "Run common subexpression elimination on graph");

    opt_module.def("fuse_operators",
        [](Graph& graph, const FusionConfig& config) {
            OperatorFusionPass pass(config);
            return pass.run(graph);
        },
        py::arg("graph"),
        py::arg("config") = FusionConfig{},
        "Run operator fusion on graph");
}
