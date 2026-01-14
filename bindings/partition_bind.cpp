#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include "pyflame_rt/partition/partition.hpp"

namespace py = pybind11;
using namespace pyflame_rt;
using namespace pyflame_rt::partition;

void bind_partition(py::module_& m) {
    auto part = m.def_submodule("partition",
        "Graph partitioning for multi-device execution");

    // ========================================================================
    // Enums
    // ========================================================================

    py::enum_<DeviceType>(part, "DeviceType",
        "Type of compute device")
        .value("CPU", DeviceType::CPU, "CPU device")
        .value("WSE", DeviceType::WSE, "Cerebras WSE")
        .value("GPU", DeviceType::GPU, "GPU device")
        .value("DISTRIBUTED", DeviceType::Distributed, "Distributed multi-node");

    py::enum_<PartitionStrategy>(part, "PartitionStrategy",
        "Strategy for partitioning graphs")
        .value("MANUAL", PartitionStrategy::Manual,
            "User-specified partitioning")
        .value("DATA_PARALLEL", PartitionStrategy::DataParallel,
            "Replicate model, split data")
        .value("MODEL_PARALLEL", PartitionStrategy::ModelParallel,
            "Split model across devices")
        .value("PIPELINE_PARALLEL", PartitionStrategy::PipelineParallel,
            "Pipeline stages across devices")
        .value("HYBRID", PartitionStrategy::Hybrid,
            "Combination of strategies")
        .value("AUTOMATIC", PartitionStrategy::Automatic,
            "Cost-model based automatic");

    py::enum_<CommType>(part, "CommType",
        "Type of communication operation")
        .value("POINT_TO_POINT", CommType::PointToPoint,
            "Direct send/recv")
        .value("ALL_REDUCE", CommType::AllReduce,
            "Reduce across all devices")
        .value("ALL_GATHER", CommType::AllGather,
            "Gather from all devices")
        .value("SCATTER", CommType::Scatter,
            "Scatter to devices")
        .value("BROADCAST", CommType::Broadcast,
            "Broadcast from one device");

    // ========================================================================
    // DeviceSpec
    // ========================================================================

    py::class_<DeviceSpec>(part, "DeviceSpec",
        "Specification of a compute device")
        .def(py::init<>())
        .def_readwrite("type", &DeviceSpec::type, "Device type")
        .def_readwrite("device_id", &DeviceSpec::device_id, "Device ID")
        .def_readwrite("memory_bytes", &DeviceSpec::memory_bytes,
            "Available memory in bytes")
        .def_readwrite("compute_flops", &DeviceSpec::compute_flops,
            "Compute capacity in FLOPS")
        .def_readwrite("memory_bandwidth", &DeviceSpec::memory_bandwidth,
            "Memory bandwidth in bytes/sec")
        .def_readwrite("name", &DeviceSpec::name, "Device name");

    // ========================================================================
    // WSEChipConfig
    // ========================================================================

    py::class_<WSEChipConfig>(part, "WSEChipConfig",
        "Configuration for multi-chip WSE")
        .def(py::init<>())
        .def_readwrite("num_chips", &WSEChipConfig::num_chips,
            "Number of chips")
        .def_readwrite("topology", &WSEChipConfig::topology,
            "Chip topology (e.g., [2, 2] for 2x2 grid)")
        .def_readwrite("inter_chip_bandwidth", &WSEChipConfig::inter_chip_bandwidth,
            "Inter-chip bandwidth in bytes/sec")
        .def_readwrite("inter_chip_latency", &WSEChipConfig::inter_chip_latency,
            "Inter-chip latency in nanoseconds")
        .def_readwrite("chip_memory_bytes", &WSEChipConfig::chip_memory_bytes,
            "Memory per chip in bytes")
        .def("total_chips", &WSEChipConfig::total_chips,
            "Get total number of chips");

    // ========================================================================
    // CommOp
    // ========================================================================

    py::class_<CommOp>(part, "CommOp",
        "Communication operation between partitions")
        .def(py::init<>())
        .def_readwrite("type", &CommOp::type, "Communication type")
        .def_readwrite("src_devices", &CommOp::src_devices, "Source devices")
        .def_readwrite("dst_devices", &CommOp::dst_devices, "Destination devices")
        .def_readwrite("tensor_name", &CommOp::tensor_name, "Tensor name")
        .def_readwrite("data_bytes", &CommOp::data_bytes, "Data size in bytes")
        .def_readwrite("estimated_latency_us", &CommOp::estimated_latency_us,
            "Estimated latency in microseconds");

    // ========================================================================
    // GraphPartition
    // ========================================================================

    py::class_<GraphPartition>(part, "GraphPartition",
        "A partition of a graph assigned to a device")
        .def(py::init<>())
        .def_readwrite("partition_id", &GraphPartition::partition_id,
            "Partition ID")
        .def_readwrite("device", &GraphPartition::device,
            "Assigned device")
        .def_readonly("nodes", &GraphPartition::nodes,
            "Nodes in this partition")
        .def_readwrite("input_tensors", &GraphPartition::input_tensors,
            "Input tensor names")
        .def_readwrite("output_tensors", &GraphPartition::output_tensors,
            "Output tensor names")
        .def_readonly("subgraph", &GraphPartition::subgraph,
            "Subgraph for this partition")
        .def_readwrite("memory_bytes", &GraphPartition::memory_bytes,
            "Memory footprint in bytes")
        .def_readwrite("compute_time_us", &GraphPartition::compute_time_us,
            "Estimated compute time in microseconds");

    // ========================================================================
    // PartitionPlan
    // ========================================================================

    py::class_<PartitionPlan>(part, "PartitionPlan",
        "Complete partitioning plan for a graph")
        .def(py::init<>())
        .def_readonly("original_graph", &PartitionPlan::original_graph,
            "Original graph")
        .def_readonly("partitions", &PartitionPlan::partitions,
            "List of partitions")
        .def_readonly("communications", &PartitionPlan::communications,
            "Communication operations")
        .def_readonly("node_to_partition", &PartitionPlan::node_to_partition,
            "Node name to partition ID mapping")
        .def_readwrite("total_latency_us", &PartitionPlan::total_latency_us,
            "Total estimated latency")
        .def_readwrite("total_comm_bytes", &PartitionPlan::total_comm_bytes,
            "Total communication bytes")
        .def_readwrite("load_imbalance", &PartitionPlan::load_imbalance,
            "Load imbalance metric");

    // ========================================================================
    // CostModel
    // ========================================================================

    py::class_<CostModel, std::shared_ptr<CostModel>>(part, "CostModel",
        "Cost model for partitioning decisions")
        .def(py::init<>())
        .def("estimate_compute_cost", &CostModel::estimate_compute_cost,
            py::arg("node"), py::arg("device"),
            "Estimate compute cost in microseconds")
        .def("estimate_memory_cost", &CostModel::estimate_memory_cost,
            py::arg("node"),
            "Estimate memory cost in bytes")
        .def("estimate_activation_memory", &CostModel::estimate_activation_memory,
            py::arg("node"),
            "Estimate activation memory in bytes")
        .def("estimate_comm_cost", &CostModel::estimate_comm_cost,
            py::arg("data_bytes"), py::arg("src"), py::arg("dst"),
            "Estimate communication cost in microseconds")
        .def("estimate_allreduce_cost", &CostModel::estimate_allreduce_cost,
            py::arg("data_bytes"), py::arg("devices"),
            "Estimate all-reduce cost in microseconds")
        .def("set_inter_device_bandwidth", &CostModel::set_inter_device_bandwidth,
            py::arg("bytes_per_sec"),
            "Set inter-device bandwidth")
        .def("set_inter_device_latency", &CostModel::set_inter_device_latency,
            py::arg("microseconds"),
            "Set inter-device latency")
        .def_property_readonly("inter_device_bandwidth",
            &CostModel::inter_device_bandwidth,
            "Get inter-device bandwidth")
        .def_property_readonly("inter_device_latency",
            &CostModel::inter_device_latency,
            "Get inter-device latency");

    // ========================================================================
    // PartitionConfig
    // ========================================================================

    py::class_<PartitionConfig>(part, "PartitionConfig",
        "Configuration for graph partitioning")
        .def(py::init<>())
        .def_readwrite("devices", &PartitionConfig::devices,
            "Target devices")
        .def_readwrite("wse_config", &PartitionConfig::wse_config,
            "WSE multi-chip configuration")
        .def_readwrite("strategy", &PartitionConfig::strategy,
            "Partitioning strategy")
        .def_readwrite("manual_assignments", &PartitionConfig::manual_assignments,
            "Manual node to device assignments")
        .def_readwrite("max_memory_per_partition",
            &PartitionConfig::max_memory_per_partition,
            "Maximum memory per partition")
        .def_readwrite("balance_compute", &PartitionConfig::balance_compute,
            "Balance compute across partitions")
        .def_readwrite("minimize_communication",
            &PartitionConfig::minimize_communication,
            "Minimize communication")
        .def_readwrite("allow_replication", &PartitionConfig::allow_replication,
            "Allow tensor replication")
        .def_readwrite("max_pipeline_stages", &PartitionConfig::max_pipeline_stages,
            "Maximum pipeline stages")
        .def_readwrite("micro_batch_size", &PartitionConfig::micro_batch_size,
            "Micro-batch size for pipeline");

    // ========================================================================
    // AnalysisResult
    // ========================================================================

    py::class_<GraphPartitioner::AnalysisResult>(part, "AnalysisResult",
        "Result of graph analysis for partitioning")
        .def_readonly("recommended_strategy",
            &GraphPartitioner::AnalysisResult::recommended_strategy,
            "Recommended partitioning strategy")
        .def_readonly("estimated_speedup",
            &GraphPartitioner::AnalysisResult::estimated_speedup,
            "Estimated speedup from partitioning")
        .def_readonly("memory_per_partition",
            &GraphPartitioner::AnalysisResult::memory_per_partition,
            "Memory per partition")
        .def_readonly("compute_per_partition",
            &GraphPartitioner::AnalysisResult::compute_per_partition,
            "Compute per partition")
        .def_readonly("communication_bytes",
            &GraphPartitioner::AnalysisResult::communication_bytes,
            "Communication volume")
        .def_readonly("bottleneck_nodes",
            &GraphPartitioner::AnalysisResult::bottleneck_nodes,
            "Bottleneck node names");

    // ========================================================================
    // PartitionStats
    // ========================================================================

    py::class_<GraphPartitioner::PartitionStats>(part, "PartitionStats",
        "Statistics about a partition plan")
        .def_readonly("num_partitions",
            &GraphPartitioner::PartitionStats::num_partitions,
            "Number of partitions")
        .def_readonly("total_nodes",
            &GraphPartitioner::PartitionStats::total_nodes,
            "Total nodes")
        .def_readonly("total_edges",
            &GraphPartitioner::PartitionStats::total_edges,
            "Total edges")
        .def_readonly("cut_edges",
            &GraphPartitioner::PartitionStats::cut_edges,
            "Edges crossing partitions")
        .def_readonly("edge_cut_ratio",
            &GraphPartitioner::PartitionStats::edge_cut_ratio,
            "Ratio of cut edges")
        .def_readonly("nodes_per_partition",
            &GraphPartitioner::PartitionStats::nodes_per_partition,
            "Nodes per partition")
        .def_readonly("memory_per_partition",
            &GraphPartitioner::PartitionStats::memory_per_partition,
            "Memory per partition");

    // ========================================================================
    // GraphPartitioner
    // ========================================================================

    py::class_<GraphPartitioner>(part, "GraphPartitioner",
        "Graph partitioner for multi-device execution")
        .def(py::init<const PartitionConfig&>(),
            py::arg("config"),
            "Create partitioner with configuration")
        .def("set_cost_model", &GraphPartitioner::set_cost_model,
            py::arg("cost_model"),
            "Set custom cost model")
        .def("partition", &GraphPartitioner::partition,
            py::arg("graph"),
            "Partition a graph")
        .def("partition_manual", &GraphPartitioner::partition_manual,
            py::arg("graph"), py::arg("assignments"),
            "Partition with manual assignments")
        .def("partition_data_parallel", &GraphPartitioner::partition_data_parallel,
            py::arg("graph"), py::arg("num_replicas"),
            "Partition for data parallelism")
        .def("partition_model_parallel", &GraphPartitioner::partition_model_parallel,
            py::arg("graph"), py::arg("num_partitions"),
            "Partition for model parallelism")
        .def("partition_pipeline", &GraphPartitioner::partition_pipeline,
            py::arg("graph"), py::arg("num_stages"),
            "Partition for pipeline parallelism")
        .def("analyze", &GraphPartitioner::analyze,
            py::arg("graph"),
            "Analyze graph for partitioning")
        .def_static("insert_communication", &GraphPartitioner::insert_communication,
            py::arg("graph"), py::arg("plan"),
            "Insert communication nodes")
        .def_static("validate", &GraphPartitioner::validate,
            py::arg("plan"),
            "Validate partition plan")
        .def_static("estimate_execution_time",
            &GraphPartitioner::estimate_execution_time,
            py::arg("plan"),
            "Estimate execution time")
        .def_static("get_stats", &GraphPartitioner::get_stats,
            py::arg("plan"),
            "Get partition statistics")
        .def_property_readonly("config", &GraphPartitioner::config,
            "Get configuration");

    // ========================================================================
    // PartitionedExecutor
    // ========================================================================

    py::class_<PartitionedExecutor>(part, "PartitionedExecutor",
        "Executor for running partitioned graphs")
        .def(py::init<const PartitionPlan&>(),
            py::arg("plan"),
            "Create executor from partition plan")
        .def("execute", &PartitionedExecutor::execute,
            py::arg("inputs"),
            "Execute partitioned graph")
        .def("execute_async", &PartitionedExecutor::execute_async,
            py::arg("inputs"),
            "Execute asynchronously")
        .def_property_readonly("plan", &PartitionedExecutor::plan,
            "Get partition plan");

    // ========================================================================
    // WSE Namespace
    // ========================================================================

    auto wse = part.def_submodule("wse",
        "WSE-specific partitioning utilities");

    wse.def("partition_for_wse", &wse::partition_for_wse,
        py::arg("graph"), py::arg("config"),
        "Partition graph for WSE multi-chip");

    wse.def("insert_wse_communication", &wse::insert_wse_communication,
        py::arg("graph"), py::arg("plan"),
        "Insert WSE-specific communication ops");

    wse.def("optimize_for_wse_dataflow", &wse::optimize_for_wse_dataflow,
        py::arg("graph"),
        "Optimize graph for WSE dataflow");

    // ========================================================================
    // Convenience Functions
    // ========================================================================

    part.def("auto_partition", &auto_partition,
        py::arg("graph"), py::arg("num_devices"),
        "Automatically partition graph");

    part.def("partition_for_multi_chip", &partition_for_multi_chip,
        py::arg("graph"), py::arg("config"),
        "Partition for multi-chip WSE");
}
