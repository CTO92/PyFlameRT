#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include "pyflame_rt/pruning/pruning.hpp"
#include "pyflame_rt/pruning/sparse_tensor.hpp"

namespace py = pybind11;
using namespace pyflame_rt;
using namespace pyflame_rt::pruning;

void bind_pruning(py::module_& m) {
    auto pruning = m.def_submodule("pruning",
        "Weight pruning for model compression");

    // ========================================================================
    // Enums
    // ========================================================================

    py::enum_<PruningGranularity>(pruning, "PruningGranularity",
        "Granularity level for pruning")
        .value("UNSTRUCTURED", PruningGranularity::Unstructured,
            "Individual weight pruning")
        .value("STRUCTURED", PruningGranularity::Structured,
            "Channel/filter pruning")
        .value("BLOCK", PruningGranularity::Block,
            "Block-sparse pruning")
        .value("NM", PruningGranularity::NM,
            "N:M sparsity pattern");

    py::enum_<PruningCriterion>(pruning, "PruningCriterion",
        "Criterion for selecting weights to prune")
        .value("MAGNITUDE", PruningCriterion::Magnitude,
            "Prune smallest magnitude weights")
        .value("MOVEMENT", PruningCriterion::Movement,
            "Movement pruning based on gradient direction")
        .value("RANDOM", PruningCriterion::Random,
            "Random pruning for baseline comparison")
        .value("TAYLOR", PruningCriterion::Taylor,
            "Taylor expansion based importance")
        .value("LAMP", PruningCriterion::LAMP,
            "Layer-Adaptive Magnitude Pruning");

    py::enum_<PruningSchedule>(pruning, "PruningSchedule",
        "Schedule for applying pruning over training")
        .value("ONE_SHOT", PruningSchedule::OneShot,
            "Prune all at once")
        .value("ITERATIVE", PruningSchedule::Iterative,
            "Gradually increase sparsity")
        .value("CUBIC", PruningSchedule::Cubic,
            "Cubic sparsity schedule")
        .value("POLYNOMIAL", PruningSchedule::Polynomial,
            "Polynomial sparsity schedule");

    py::enum_<SparseFormat>(pruning, "SparseFormat",
        "Sparse tensor storage format")
        .value("COO", SparseFormat::COO,
            "Coordinate format")
        .value("CSR", SparseFormat::CSR,
            "Compressed Sparse Row")
        .value("CSC", SparseFormat::CSC,
            "Compressed Sparse Column")
        .value("BSR", SparseFormat::BSR,
            "Block Sparse Row");

    // ========================================================================
    // PruningConfig
    // ========================================================================

    py::class_<PruningConfig>(pruning, "PruningConfig",
        "Configuration for weight pruning")
        .def(py::init<>())
        .def_readwrite("target_sparsity", &PruningConfig::target_sparsity,
            "Target sparsity level (0.0 to 1.0)")
        .def_readwrite("granularity", &PruningConfig::granularity,
            "Pruning granularity")
        .def_readwrite("criterion", &PruningConfig::criterion,
            "Weight importance criterion")
        .def_readwrite("schedule", &PruningConfig::schedule,
            "Pruning schedule")
        .def_readwrite("n_value", &PruningConfig::n_value,
            "N value for N:M sparsity")
        .def_readwrite("m_value", &PruningConfig::m_value,
            "M value for N:M sparsity")
        .def_readwrite("block_size", &PruningConfig::block_size,
            "Block size for block-sparse pruning")
        .def_readwrite("start_step", &PruningConfig::start_step,
            "Step to start pruning")
        .def_readwrite("end_step", &PruningConfig::end_step,
            "Step to end pruning")
        .def_readwrite("pruning_frequency", &PruningConfig::pruning_frequency,
            "Frequency of pruning updates")
        .def_readwrite("initial_sparsity", &PruningConfig::initial_sparsity,
            "Initial sparsity before gradual pruning")
        .def_readwrite("per_layer_sparsity", &PruningConfig::per_layer_sparsity,
            "Per-layer sparsity targets")
        .def("add_exclude_layer", [](PruningConfig& self, const std::string& layer) {
            self.exclude_layers.insert(layer);
        }, "Add layer name to exclude from pruning")
        .def_static("magnitude_pruning", &PruningConfig::magnitude_pruning,
            py::arg("sparsity"),
            "Create magnitude-based pruning config")
        .def_static("structured_pruning", &PruningConfig::structured_pruning,
            py::arg("sparsity"),
            "Create structured (channel) pruning config")
        .def_static("nm_sparsity", &PruningConfig::nm_sparsity,
            py::arg("n"), py::arg("m"),
            "Create N:M sparsity config");

    // ========================================================================
    // PruningStats
    // ========================================================================

    py::class_<PruningStats>(pruning, "PruningStats",
        "Statistics from pruning operation")
        .def(py::init<>())
        .def_readonly("total_params", &PruningStats::total_params,
            "Total number of parameters")
        .def_readonly("pruned_params", &PruningStats::pruned_params,
            "Number of pruned parameters")
        .def_readonly("original_size_bytes", &PruningStats::original_size_bytes,
            "Original model size in bytes")
        .def_readonly("pruned_size_bytes", &PruningStats::pruned_size_bytes,
            "Pruned model size in bytes")
        .def_readonly("per_layer_sparsity", &PruningStats::per_layer_sparsity,
            "Sparsity per layer")
        .def("actual_sparsity", &PruningStats::actual_sparsity,
            "Get actual achieved sparsity")
        .def("compression_ratio", &PruningStats::compression_ratio,
            "Get compression ratio");

    // ========================================================================
    // PruningMask
    // ========================================================================

    py::class_<PruningMask>(pruning, "PruningMask",
        "Binary mask indicating pruned weights")
        .def(py::init<>())
        .def_property_readonly("shape", &PruningMask::shape,
            "Shape of the mask")
        .def("sparsity", &PruningMask::sparsity,
            "Get sparsity of the mask")
        .def("num_zeros", &PruningMask::num_zeros,
            "Get number of zeros")
        .def("num_nonzeros", &PruningMask::num_nonzeros,
            "Get number of nonzeros")
        .def("apply", &PruningMask::apply,
            py::arg("tensor"),
            "Apply mask to tensor")
        .def_static("from_tensor", &PruningMask::from_tensor,
            py::arg("tensor"), py::arg("threshold") = 0.0f,
            "Create mask from tensor values");

    // ========================================================================
    // WeightPruner
    // ========================================================================

    py::class_<WeightPruner>(pruning, "WeightPruner",
        "Weight pruner for model compression")
        .def(py::init<const PruningConfig&>(),
            py::arg("config"),
            "Create pruner with configuration")
        .def("prune", &WeightPruner::prune,
            py::arg("graph"),
            "Prune weights in a graph")
        .def("compute_masks", &WeightPruner::compute_masks,
            py::arg("graph"),
            "Compute pruning masks without applying")
        .def("apply_masks", &WeightPruner::apply_masks,
            py::arg("graph"), py::arg("masks"),
            "Apply pre-computed masks to graph")
        .def("get_sparsity_at_step", &WeightPruner::get_sparsity_at_step,
            py::arg("step"),
            "Get target sparsity at given step")
        .def("get_stats", &WeightPruner::get_stats,
            "Get pruning statistics")
        .def_property_readonly("config",
            [](const WeightPruner& p) { return p.config(); },
            "Get pruner configuration");

    // ========================================================================
    // SparseTensor
    // ========================================================================

    py::class_<SparseTensor>(pruning, "SparseTensor",
        "Sparse tensor representation")
        .def(py::init<>())
        .def_property_readonly("format", &SparseTensor::format,
            "Sparse format")
        .def_property_readonly("shape", &SparseTensor::shape,
            "Dense shape")
        .def_property_readonly("dtype", &SparseTensor::dtype,
            "Data type")
        .def("nnz", &SparseTensor::nnz,
            "Number of non-zeros")
        .def("sparsity", &SparseTensor::sparsity,
            "Sparsity ratio")
        .def("to_dense", &SparseTensor::to_dense,
            "Convert to dense tensor")
        .def("to_coo", &SparseTensor::to_coo,
            "Convert to COO format")
        .def("to_csr", &SparseTensor::to_csr,
            "Convert to CSR format")
        .def("to_csc", &SparseTensor::to_csc,
            "Convert to CSC format")
        .def("memory_bytes", &SparseTensor::memory_bytes,
            "Memory usage in bytes")
        .def("compression_ratio", &SparseTensor::compression_ratio,
            "Compression ratio vs dense")
        .def_static("from_dense", &SparseTensor::from_dense,
            py::arg("tensor"), py::arg("format") = SparseFormat::COO,
            "Create sparse tensor from dense")
        .def_static("from_coo", &SparseTensor::from_coo,
            py::arg("shape"), py::arg("indices"), py::arg("values"), py::arg("dtype"),
            "Create from COO data");

    // ========================================================================
    // Sparse Operations
    // ========================================================================

    pruning.def("sparse_matmul", &sparse_matmul,
        py::arg("sparse"), py::arg("dense"),
        "Sparse-dense matrix multiplication");

    pruning.def("sparse_add", &sparse_add,
        py::arg("a"), py::arg("b"),
        "Element-wise addition of sparse tensors");

    // ========================================================================
    // Utility Functions
    // ========================================================================

    pruning.def("magnitude_prune", &magnitude_prune,
        py::arg("graph"), py::arg("sparsity"),
        "Convenience function for magnitude pruning");

    pruning.def("structured_prune", &structured_prune,
        py::arg("graph"), py::arg("sparsity"),
        "Convenience function for structured pruning");

    pruning.def("nm_sparsity_prune", &nm_sparsity_prune,
        py::arg("graph"), py::arg("n"), py::arg("m"),
        "Convenience function for N:M sparsity");

    pruning.def("analyze_sparsity", &analyze_sparsity,
        py::arg("graph"),
        "Analyze current sparsity of a graph");
}
