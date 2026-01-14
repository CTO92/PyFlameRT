#include <gtest/gtest.h>
#include "pyflame_rt/graph.hpp"
#include "pyflame_rt/errors.hpp"

using namespace pyflame_rt;

TEST(NodeTest, Construction) {
    Node node("add_0", "Add", {"a", "b"}, {"c"});

    EXPECT_EQ(node.name(), "add_0");
    EXPECT_EQ(node.op_type(), "Add");
    EXPECT_EQ(node.inputs().size(), 2);
    EXPECT_EQ(node.outputs().size(), 1);
}

TEST(NodeTest, Attributes) {
    Node node("node", "Op", {}, {});

    node.set_attr("alpha", 0.5f);
    node.set_attr("axis", int64_t(1));
    node.set_attr("name", std::string("test"));

    EXPECT_TRUE(node.has_attr("alpha"));
    EXPECT_FALSE(node.has_attr("beta"));

    EXPECT_FLOAT_EQ(node.get_attr<float>("alpha", 0.0f), 0.5f);
    EXPECT_EQ(node.get_attr<int64_t>("axis", 0), 1);
    EXPECT_EQ(node.get_attr<std::string>("name", ""), "test");

    // Default value for missing attribute
    EXPECT_FLOAT_EQ(node.get_attr<float>("missing", 1.0f), 1.0f);
}

TEST(GraphTest, Empty) {
    Graph graph("test_graph");

    EXPECT_EQ(graph.name(), "test_graph");
    EXPECT_EQ(graph.num_nodes(), 0);
    EXPECT_TRUE(graph.inputs().empty());
    EXPECT_TRUE(graph.outputs().empty());
}

TEST(GraphTest, AddNodes) {
    Graph graph;

    auto node1 = std::make_shared<Node>("add_0", "Add", std::vector<std::string>{"a", "b"}, std::vector<std::string>{"c"});
    auto node2 = std::make_shared<Node>("relu_0", "Relu", std::vector<std::string>{"c"}, std::vector<std::string>{"d"});

    graph.add_node(node1);
    graph.add_node(node2);

    EXPECT_EQ(graph.num_nodes(), 2);
    EXPECT_EQ(graph.get_node("add_0"), node1.get());
    EXPECT_EQ(graph.get_node("relu_0"), node2.get());
    EXPECT_EQ(graph.get_node("nonexistent"), nullptr);
}

TEST(GraphTest, DuplicateNode) {
    Graph graph;

    auto node1 = std::make_shared<Node>("node", "Op", std::vector<std::string>{}, std::vector<std::string>{});
    auto node2 = std::make_shared<Node>("node", "Op", std::vector<std::string>{}, std::vector<std::string>{});

    graph.add_node(node1);
    EXPECT_THROW(graph.add_node(node2), std::invalid_argument);
}

TEST(GraphTest, InputsOutputs) {
    Graph graph;

    TensorInfo input("input", {{1}, {3}, {224}, {224}}, DType::Float32);
    TensorInfo output("output", {{1}, {1000}}, DType::Float32);

    graph.add_input(input);
    graph.add_output(output);

    EXPECT_EQ(graph.inputs().size(), 1);
    EXPECT_EQ(graph.outputs().size(), 1);
    EXPECT_EQ(graph.inputs()[0].name, "input");
    EXPECT_EQ(graph.outputs()[0].name, "output");
}

TEST(GraphTest, Initializers) {
    Graph graph;

    Tensor weights({3, 3}, DType::Float32);
    weights.fill(1.0f);

    graph.add_initializer("weight", std::move(weights));

    EXPECT_TRUE(graph.has_initializer("weight"));
    EXPECT_FALSE(graph.has_initializer("bias"));

    const Tensor* w = graph.get_initializer("weight");
    EXPECT_NE(w, nullptr);
    EXPECT_EQ(w->num_elements(), 9);
}

TEST(GraphTest, TopologicalSort) {
    Graph graph;

    // Create a simple linear graph: a -> b -> c
    auto node1 = std::make_shared<Node>("n1", "Op", std::vector<std::string>{"input"}, std::vector<std::string>{"a"});
    auto node2 = std::make_shared<Node>("n2", "Op", std::vector<std::string>{"a"}, std::vector<std::string>{"b"});
    auto node3 = std::make_shared<Node>("n3", "Op", std::vector<std::string>{"b"}, std::vector<std::string>{"output"});

    // Add in reverse order
    graph.add_node(node3);
    graph.add_node(node1);
    graph.add_node(node2);

    auto sorted = graph.topological_sort();

    EXPECT_EQ(sorted.size(), 3);
    EXPECT_EQ(sorted[0]->name(), "n1");
    EXPECT_EQ(sorted[1]->name(), "n2");
    EXPECT_EQ(sorted[2]->name(), "n3");
}

TEST(GraphTest, Validation) {
    Graph graph;

    TensorInfo input("input", {{1}}, DType::Float32);
    TensorInfo output("output", {{1}}, DType::Float32);

    graph.add_input(input);
    graph.add_output(output);

    auto node = std::make_shared<Node>("n1", "Op", std::vector<std::string>{"input"}, std::vector<std::string>{"output"});
    graph.add_node(node);

    auto errors = graph.validate();
    EXPECT_TRUE(errors.empty());
}

TEST(GraphTest, ValidationMissingInput) {
    Graph graph;

    TensorInfo output("output", {{1}}, DType::Float32);
    graph.add_output(output);

    // Node references input that doesn't exist
    auto node = std::make_shared<Node>("n1", "Op", std::vector<std::string>{"missing_input"}, std::vector<std::string>{"output"});
    graph.add_node(node);

    auto errors = graph.validate();
    EXPECT_FALSE(errors.empty());
}

TEST(GraphTest, ProducerConsumer) {
    Graph graph;

    auto node1 = std::make_shared<Node>("n1", "Op", std::vector<std::string>{}, std::vector<std::string>{"a"});
    auto node2 = std::make_shared<Node>("n2", "Op", std::vector<std::string>{"a"}, std::vector<std::string>{"b"});

    graph.add_node(node1);
    graph.add_node(node2);

    const std::string* producer = graph.get_producer("a");
    EXPECT_NE(producer, nullptr);
    EXPECT_EQ(*producer, "n1");

    auto consumers = graph.get_consumers("a");
    EXPECT_EQ(consumers.size(), 1);
    EXPECT_EQ(consumers[0], "n2");
}
