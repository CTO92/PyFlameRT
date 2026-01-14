#include <gtest/gtest.h>
#include "pyflame_rt/partition/partition.hpp"
#include "pyflame_rt/graph.hpp"
#include "pyflame_rt/node.hpp"

using namespace pyflame_rt;
using namespace pyflame_rt::partition;

// ============================================================================
// DeviceSpec Tests
// ============================================================================

TEST(DeviceSpecTest, DefaultConstruction) {
    DeviceSpec device;
    EXPECT_EQ(device.type, DeviceType::CPU);
    EXPECT_EQ(device.device_id, 0);
    EXPECT_EQ(device.memory_bytes, 0);
    EXPECT_EQ(device.compute_flops, 0.0);
}

TEST(DeviceSpecTest, CustomDevice) {
    DeviceSpec device;
    device.type = DeviceType::WSE;
    device.device_id = 1;
    device.memory_bytes = 40ULL * 1024 * 1024 * 1024;  // 40GB
    device.compute_flops = 1e15;  // 1 PFLOP
    device.name = "WSE-2";

    EXPECT_EQ(device.type, DeviceType::WSE);
    EXPECT_EQ(device.device_id, 1);
    EXPECT_EQ(device.name, "WSE-2");
}

// ============================================================================
// WSEChipConfig Tests
// ============================================================================

TEST(WSEChipConfigTest, DefaultConstruction) {
    WSEChipConfig config;
    EXPECT_EQ(config.num_chips, 1);
    EXPECT_EQ(config.total_chips(), 1);
}

TEST(WSEChipConfigTest, GridTopology) {
    WSEChipConfig config;
    config.topology = {2, 2};
    EXPECT_EQ(config.total_chips(), 4);

    config.topology = {2, 3};
    EXPECT_EQ(config.total_chips(), 6);

    config.topology = {1, 4};
    EXPECT_EQ(config.total_chips(), 4);
}

TEST(WSEChipConfigTest, ChipConfiguration) {
    WSEChipConfig config;
    config.num_chips = 4;
    config.topology = {2, 2};
    config.inter_chip_bandwidth = 1e12;  // 1 TB/s
    config.inter_chip_latency = 100.0;   // 100 ns
    config.chip_memory_bytes = 40ULL * 1024 * 1024 * 1024;

    EXPECT_EQ(config.total_chips(), 4);
    EXPECT_EQ(config.chip_memory_bytes, 40ULL * 1024 * 1024 * 1024);
}

// ============================================================================
// PartitionConfig Tests
// ============================================================================

TEST(PartitionConfigTest, DefaultConstruction) {
    PartitionConfig config;
    EXPECT_EQ(config.strategy, PartitionStrategy::Automatic);
    EXPECT_TRUE(config.balance_compute);
    EXPECT_TRUE(config.minimize_communication);
    EXPECT_FALSE(config.allow_replication);
}

TEST(PartitionConfigTest, WithDevices) {
    PartitionConfig config;

    for (int i = 0; i < 4; ++i) {
        DeviceSpec device;
        device.type = DeviceType::CPU;
        device.device_id = i;
        config.devices.push_back(device);
    }

    config.strategy = PartitionStrategy::ModelParallel;

    EXPECT_EQ(config.devices.size(), 4);
    EXPECT_EQ(config.strategy, PartitionStrategy::ModelParallel);
}

// ============================================================================
// CostModel Tests
// ============================================================================

TEST(CostModelTest, DefaultConstruction) {
    CostModel model;
    EXPECT_GT(model.inter_device_bandwidth(), 0.0);
    EXPECT_GT(model.inter_device_latency(), 0.0);
}

TEST(CostModelTest, SetBandwidth) {
    CostModel model;
    model.set_inter_device_bandwidth(1e10);  // 10 GB/s
    EXPECT_DOUBLE_EQ(model.inter_device_bandwidth(), 1e10);
}

TEST(CostModelTest, SetLatency) {
    CostModel model;
    model.set_inter_device_latency(5.0);  // 5 microseconds
    EXPECT_DOUBLE_EQ(model.inter_device_latency(), 5.0);
}

TEST(CostModelTest, EstimateCommCostSameDevice) {
    CostModel model;

    DeviceSpec device;
    device.device_id = 0;

    // Communication on same device should be zero
    double cost = model.estimate_comm_cost(1024, device, device);
    EXPECT_DOUBLE_EQ(cost, 0.0);
}

TEST(CostModelTest, EstimateCommCostDifferentDevices) {
    CostModel model;
    model.set_inter_device_bandwidth(1e9);  // 1 GB/s
    model.set_inter_device_latency(10.0);   // 10 us

    DeviceSpec src, dst;
    src.device_id = 0;
    dst.device_id = 1;

    double cost = model.estimate_comm_cost(1024 * 1024, src, dst);  // 1MB transfer

    // Cost = latency + data_bytes / bandwidth
    // Expected ~= 10 us + 1MB / (1 GB/s) = 10 + 1000 = 1010 us
    EXPECT_GT(cost, 10.0);
    EXPECT_LT(cost, 2000.0);
}

TEST(CostModelTest, EstimateAllReduceCost) {
    CostModel model;

    std::vector<DeviceSpec> devices;
    for (int i = 0; i < 4; ++i) {
        DeviceSpec device;
        device.device_id = i;
        devices.push_back(device);
    }

    double cost = model.estimate_allreduce_cost(1024 * 1024, devices);

    // Should be non-zero for multiple devices
    EXPECT_GT(cost, 0.0);
}

TEST(CostModelTest, EstimateAllReduceSingleDevice) {
    CostModel model;

    std::vector<DeviceSpec> devices;
    DeviceSpec device;
    device.device_id = 0;
    devices.push_back(device);

    double cost = model.estimate_allreduce_cost(1024 * 1024, devices);

    // Single device all-reduce should be zero
    EXPECT_DOUBLE_EQ(cost, 0.0);
}

// ============================================================================
// GraphPartitioner Tests
// ============================================================================

class GraphPartitionerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a simple test graph
        graph_ = std::make_unique<Graph>();

        // Add input
        NodeArg input_arg;
        input_arg.name = "input";
        input_arg.dtype = DType::Float32;
        input_arg.shape = {1, 10};
        graph_->add_input(input_arg);

        // Add some nodes
        auto node1 = std::make_shared<Node>("MatMul", "matmul_1");
        node1->add_input(input_arg);
        NodeArg out1;
        out1.name = "matmul_1_out";
        out1.dtype = DType::Float32;
        out1.shape = {1, 10};
        node1->add_output(out1);
        graph_->add_node(node1);

        auto node2 = std::make_shared<Node>("Relu", "relu_1");
        node2->add_input(out1);
        NodeArg out2;
        out2.name = "relu_1_out";
        out2.dtype = DType::Float32;
        out2.shape = {1, 10};
        node2->add_output(out2);
        graph_->add_node(node2);

        auto node3 = std::make_shared<Node>("MatMul", "matmul_2");
        node3->add_input(out2);
        NodeArg out3;
        out3.name = "output";
        out3.dtype = DType::Float32;
        out3.shape = {1, 5};
        node3->add_output(out3);
        graph_->add_node(node3);

        graph_->add_output(out3);

        // Create config with 2 devices
        config_.strategy = PartitionStrategy::ModelParallel;
        for (int i = 0; i < 2; ++i) {
            DeviceSpec device;
            device.type = DeviceType::CPU;
            device.device_id = i;
            device.memory_bytes = 1024 * 1024 * 1024;  // 1GB
            config_.devices.push_back(device);
        }
    }

    std::unique_ptr<Graph> graph_;
    PartitionConfig config_;
};

TEST_F(GraphPartitionerTest, Construction) {
    GraphPartitioner partitioner(config_);
    EXPECT_EQ(partitioner.config().strategy, PartitionStrategy::ModelParallel);
}

TEST_F(GraphPartitionerTest, PartitionModelParallel) {
    GraphPartitioner partitioner(config_);
    auto plan = partitioner.partition_model_parallel(*graph_, 2);

    EXPECT_EQ(plan.partitions.size(), 2);
    EXPECT_NE(plan.original_graph, nullptr);
}

TEST_F(GraphPartitionerTest, PartitionDataParallel) {
    GraphPartitioner partitioner(config_);
    auto plan = partitioner.partition_data_parallel(*graph_, 2);

    EXPECT_EQ(plan.partitions.size(), 2);
    // In data parallel, each partition should have all nodes
    for (const auto& partition : plan.partitions) {
        EXPECT_EQ(partition.nodes.size(), graph_->nodes().size());
    }
}

TEST_F(GraphPartitionerTest, PartitionPipeline) {
    GraphPartitioner partitioner(config_);
    auto plan = partitioner.partition_pipeline(*graph_, 2);

    EXPECT_EQ(plan.partitions.size(), 2);
    // Total nodes across partitions should equal graph nodes
    size_t total_nodes = 0;
    for (const auto& partition : plan.partitions) {
        total_nodes += partition.nodes.size();
    }
    EXPECT_EQ(total_nodes, graph_->nodes().size());
}

TEST_F(GraphPartitionerTest, Validate) {
    GraphPartitioner partitioner(config_);
    auto plan = partitioner.partition_model_parallel(*graph_, 2);

    EXPECT_TRUE(GraphPartitioner::validate(plan));
}

TEST_F(GraphPartitionerTest, GetStats) {
    GraphPartitioner partitioner(config_);
    auto plan = partitioner.partition_model_parallel(*graph_, 2);

    auto stats = GraphPartitioner::get_stats(plan);

    EXPECT_EQ(stats.num_partitions, 2);
    EXPECT_EQ(stats.total_nodes, graph_->nodes().size());
}

TEST_F(GraphPartitionerTest, Analyze) {
    GraphPartitioner partitioner(config_);
    auto analysis = partitioner.analyze(*graph_);

    // Should recommend some strategy
    EXPECT_TRUE(
        analysis.recommended_strategy == PartitionStrategy::DataParallel ||
        analysis.recommended_strategy == PartitionStrategy::ModelParallel ||
        analysis.recommended_strategy == PartitionStrategy::PipelineParallel
    );
    EXPECT_GE(analysis.estimated_speedup, 1.0);
}

// ============================================================================
// PartitionPlan Tests
// ============================================================================

TEST(PartitionPlanTest, DefaultConstruction) {
    PartitionPlan plan;
    EXPECT_TRUE(plan.partitions.empty());
    EXPECT_TRUE(plan.communications.empty());
    EXPECT_EQ(plan.total_latency_us, 0.0);
    EXPECT_EQ(plan.total_comm_bytes, 0);
}

// ============================================================================
// CommOp Tests
// ============================================================================

TEST(CommOpTest, DefaultConstruction) {
    CommOp op;
    EXPECT_TRUE(op.src_devices.empty());
    EXPECT_TRUE(op.dst_devices.empty());
    EXPECT_TRUE(op.tensor_name.empty());
    EXPECT_EQ(op.data_bytes, 0);
}

TEST(CommOpTest, PointToPoint) {
    CommOp op;
    op.type = CommType::PointToPoint;
    op.src_devices = {0};
    op.dst_devices = {1};
    op.tensor_name = "activation";
    op.data_bytes = 1024 * 1024;

    EXPECT_EQ(op.type, CommType::PointToPoint);
    EXPECT_EQ(op.src_devices.size(), 1);
    EXPECT_EQ(op.dst_devices.size(), 1);
}

TEST(CommOpTest, AllReduce) {
    CommOp op;
    op.type = CommType::AllReduce;
    op.src_devices = {0, 1, 2, 3};
    op.dst_devices = {0, 1, 2, 3};
    op.tensor_name = "gradients";
    op.data_bytes = 4 * 1024 * 1024;

    EXPECT_EQ(op.type, CommType::AllReduce);
    EXPECT_EQ(op.src_devices.size(), 4);
}

// ============================================================================
// Convenience Functions Tests
// ============================================================================

TEST_F(GraphPartitionerTest, AutoPartition) {
    auto plan = auto_partition(*graph_, 2);

    EXPECT_EQ(plan.partitions.size(), 2);
    EXPECT_TRUE(GraphPartitioner::validate(plan));
}
