#include <gtest/gtest.h>
#include "pyflame_rt/graph.hpp"
#include "io/loader.hpp"
#include "io/pyflame_format.hpp"
#include <filesystem>
#include <fstream>

using namespace pyflame_rt;
namespace fs = std::filesystem;

class LoaderTest : public ::testing::Test {
protected:
    fs::path temp_dir;

    void SetUp() override {
        temp_dir = fs::temp_directory_path() / "pyflame_rt_test";
        fs::create_directories(temp_dir);
    }

    void TearDown() override {
        fs::remove_all(temp_dir);
    }
};

TEST_F(LoaderTest, GetExtension) {
    EXPECT_EQ(get_extension("model.pfm"), ".pfm");
    EXPECT_EQ(get_extension("path/to/model.PFM"), ".pfm");
    EXPECT_EQ(get_extension("no_extension"), "");
    EXPECT_EQ(get_extension("file.onnx"), ".onnx");
}

TEST_F(LoaderTest, UnsupportedFormat) {
    fs::path model_path = temp_dir / "model.onnx";
    std::ofstream(model_path).close();

    EXPECT_THROW(load_model(model_path.string()), UnsupportedFormatError);
}

TEST_F(LoaderTest, FileNotFound) {
    fs::path model_path = temp_dir / "nonexistent.pfm";

    EXPECT_THROW(load_model(model_path.string()), InvalidModelError);
}

TEST_F(LoaderTest, InvalidMagic) {
    fs::path model_path = temp_dir / "invalid.pfm";

    std::ofstream file(model_path, std::ios::binary);
    file.write("XXXX", 4);  // Invalid magic
    file.close();

    EXPECT_THROW(load_model(model_path.string()), InvalidModelError);
}

TEST_F(LoaderTest, RoundTrip) {
    // Create a simple graph
    auto graph = std::make_unique<Graph>("test_graph");

    TensorInfo input("input", {{1}, {3}, {224}, {224}}, DType::Float32);
    TensorInfo output("output", {{1}, {1000}}, DType::Float32);

    graph->add_input(input);
    graph->add_output(output);

    // Add a weight initializer
    Tensor weight({1000, 2048}, DType::Float32);
    weight.fill(0.01f);
    graph->add_initializer("fc_weight", std::move(weight));

    // Add nodes
    auto relu = std::make_shared<Node>(
        "relu_0", "Relu",
        std::vector<std::string>{"input"},
        std::vector<std::string>{"relu_out"}
    );
    relu->set_attr("custom_attr", 42.0f);
    graph->add_node(relu);

    auto flatten = std::make_shared<Node>(
        "flatten_0", "Flatten",
        std::vector<std::string>{"relu_out"},
        std::vector<std::string>{"flat_out"}
    );
    flatten->set_attr("axis", int64_t(1));
    graph->add_node(flatten);

    // Save to file
    fs::path model_path = temp_dir / "test_model.pfm";
    save_model(*graph, model_path.string());

    EXPECT_TRUE(fs::exists(model_path));

    // Load back
    auto loaded = load_model(model_path.string());

    EXPECT_EQ(loaded->name(), "test_graph");
    EXPECT_EQ(loaded->inputs().size(), 1);
    EXPECT_EQ(loaded->outputs().size(), 1);
    EXPECT_EQ(loaded->num_nodes(), 2);

    // Check input
    EXPECT_EQ(loaded->inputs()[0].name, "input");
    EXPECT_EQ(loaded->inputs()[0].dtype, DType::Float32);

    // Check initializer
    EXPECT_TRUE(loaded->has_initializer("fc_weight"));
    const Tensor* loaded_weight = loaded->get_initializer("fc_weight");
    EXPECT_EQ(loaded_weight->shape()[0], 1000);
    EXPECT_EQ(loaded_weight->shape()[1], 2048);

    // Check nodes
    const Node* relu_loaded = loaded->get_node("relu_0");
    ASSERT_NE(relu_loaded, nullptr);
    EXPECT_EQ(relu_loaded->op_type(), "Relu");
    EXPECT_FLOAT_EQ(relu_loaded->get_attr<float>("custom_attr", 0.0f), 42.0f);

    const Node* flatten_loaded = loaded->get_node("flatten_0");
    ASSERT_NE(flatten_loaded, nullptr);
    EXPECT_EQ(flatten_loaded->get_attr<int64_t>("axis", 0), 1);
}
