#include <gtest/gtest.h>
#include "pyflame_rt/graph.hpp"
#include "pyflame_rt/backend.hpp"
#include "backends/cpu/executor.hpp"
#include <cmath>

using namespace pyflame_rt;

class ExecutorTest : public ::testing::Test {
protected:
    std::unique_ptr<CPUExecutor> executor;

    void SetUp() override {
        executor = std::make_unique<CPUExecutor>();
    }
};

TEST_F(ExecutorTest, BasicExecution) {
    // Create a simple graph: output = relu(input)
    Graph graph("test");

    TensorInfo input_info("input", {{2}, {3}}, DType::Float32);
    TensorInfo output_info("output", {{2}, {3}}, DType::Float32);

    graph.add_input(input_info);
    graph.add_output(output_info);

    auto relu_node = std::make_shared<Node>(
        "relu_0", "Relu",
        std::vector<std::string>{"input"},
        std::vector<std::string>{"output"}
    );
    graph.add_node(relu_node);

    // Create input tensor
    Tensor input({2, 3}, DType::Float32);
    float* data = input.data_ptr<float>();
    data[0] = -1.0f;
    data[1] = 0.0f;
    data[2] = 1.0f;
    data[3] = -2.0f;
    data[4] = 2.0f;
    data[5] = 3.0f;

    std::unordered_map<std::string, Tensor> feeds;
    feeds["input"] = std::move(input);

    auto outputs = executor->execute(graph, feeds);

    ASSERT_EQ(outputs.size(), 1);
    const float* out = outputs[0].data_ptr<float>();

    EXPECT_FLOAT_EQ(out[0], 0.0f);   // relu(-1)
    EXPECT_FLOAT_EQ(out[1], 0.0f);   // relu(0)
    EXPECT_FLOAT_EQ(out[2], 1.0f);   // relu(1)
    EXPECT_FLOAT_EQ(out[3], 0.0f);   // relu(-2)
    EXPECT_FLOAT_EQ(out[4], 2.0f);   // relu(2)
    EXPECT_FLOAT_EQ(out[5], 3.0f);   // relu(3)
}

TEST_F(ExecutorTest, MultiNodeExecution) {
    // Create graph: c = add(a, b), d = relu(c)
    Graph graph("test");

    TensorInfo a_info("a", {{4}}, DType::Float32);
    TensorInfo b_info("b", {{4}}, DType::Float32);
    TensorInfo d_info("d", {{4}}, DType::Float32);

    graph.add_input(a_info);
    graph.add_input(b_info);
    graph.add_output(d_info);

    auto add_node = std::make_shared<Node>(
        "add_0", "Add",
        std::vector<std::string>{"a", "b"},
        std::vector<std::string>{"c"}
    );
    auto relu_node = std::make_shared<Node>(
        "relu_0", "Relu",
        std::vector<std::string>{"c"},
        std::vector<std::string>{"d"}
    );

    graph.add_node(add_node);
    graph.add_node(relu_node);

    // Create inputs
    Tensor a({4}, DType::Float32);
    Tensor b({4}, DType::Float32);

    float* a_data = a.data_ptr<float>();
    float* b_data = b.data_ptr<float>();

    a_data[0] = 1.0f;  a_data[1] = -2.0f;  a_data[2] = 3.0f;  a_data[3] = -4.0f;
    b_data[0] = -2.0f; b_data[1] = 1.0f;   b_data[2] = -1.0f; b_data[3] = 5.0f;

    std::unordered_map<std::string, Tensor> feeds;
    feeds["a"] = std::move(a);
    feeds["b"] = std::move(b);

    auto outputs = executor->execute(graph, feeds);

    ASSERT_EQ(outputs.size(), 1);
    const float* out = outputs[0].data_ptr<float>();

    // a + b = [-1, -1, 2, 1], relu = [0, 0, 2, 1]
    EXPECT_FLOAT_EQ(out[0], 0.0f);
    EXPECT_FLOAT_EQ(out[1], 0.0f);
    EXPECT_FLOAT_EQ(out[2], 2.0f);
    EXPECT_FLOAT_EQ(out[3], 1.0f);
}

TEST_F(ExecutorTest, WithInitializers) {
    // Create graph: output = add(input, weight)
    Graph graph("test");

    TensorInfo input_info("input", {{3}}, DType::Float32);
    TensorInfo output_info("output", {{3}}, DType::Float32);

    graph.add_input(input_info);
    graph.add_output(output_info);

    // Add weight as initializer
    Tensor weight({3}, DType::Float32);
    weight.fill(2.0f);
    graph.add_initializer("weight", std::move(weight));

    auto add_node = std::make_shared<Node>(
        "add_0", "Add",
        std::vector<std::string>{"input", "weight"},
        std::vector<std::string>{"output"}
    );
    graph.add_node(add_node);

    // Create input
    Tensor input({3}, DType::Float32);
    input.fill(1.0f);

    std::unordered_map<std::string, Tensor> feeds;
    feeds["input"] = std::move(input);

    auto outputs = executor->execute(graph, feeds);

    ASSERT_EQ(outputs.size(), 1);
    const float* out = outputs[0].data_ptr<float>();

    EXPECT_FLOAT_EQ(out[0], 3.0f);
    EXPECT_FLOAT_EQ(out[1], 3.0f);
    EXPECT_FLOAT_EQ(out[2], 3.0f);
}

TEST_F(ExecutorTest, SupportedOps) {
    auto ops = executor->get_supported_ops();

    // Check some basic ops are registered
    EXPECT_TRUE(executor->supports_op("Add"));
    EXPECT_TRUE(executor->supports_op("Relu"));
    EXPECT_TRUE(executor->supports_op("MatMul"));
    EXPECT_FALSE(executor->supports_op("NonExistentOp"));
}

TEST_F(ExecutorTest, MatMul) {
    Graph graph("test");

    TensorInfo a_info("a", {{2}, {3}}, DType::Float32);
    TensorInfo b_info("b", {{3}, {2}}, DType::Float32);
    TensorInfo c_info("c", {{2}, {2}}, DType::Float32);

    graph.add_input(a_info);
    graph.add_input(b_info);
    graph.add_output(c_info);

    auto matmul = std::make_shared<Node>(
        "matmul_0", "MatMul",
        std::vector<std::string>{"a", "b"},
        std::vector<std::string>{"c"}
    );
    graph.add_node(matmul);

    Tensor a({2, 3}, DType::Float32);
    Tensor b({3, 2}, DType::Float32);

    // a = [[1, 2, 3], [4, 5, 6]]
    float* a_data = a.data_ptr<float>();
    a_data[0] = 1; a_data[1] = 2; a_data[2] = 3;
    a_data[3] = 4; a_data[4] = 5; a_data[5] = 6;

    // b = [[1, 2], [3, 4], [5, 6]]
    float* b_data = b.data_ptr<float>();
    b_data[0] = 1; b_data[1] = 2;
    b_data[2] = 3; b_data[3] = 4;
    b_data[4] = 5; b_data[5] = 6;

    std::unordered_map<std::string, Tensor> feeds;
    feeds["a"] = std::move(a);
    feeds["b"] = std::move(b);

    auto outputs = executor->execute(graph, feeds);

    ASSERT_EQ(outputs.size(), 1);
    const float* out = outputs[0].data_ptr<float>();

    // c = [[22, 28], [49, 64]]
    EXPECT_FLOAT_EQ(out[0], 22.0f);
    EXPECT_FLOAT_EQ(out[1], 28.0f);
    EXPECT_FLOAT_EQ(out[2], 49.0f);
    EXPECT_FLOAT_EQ(out[3], 64.0f);
}
