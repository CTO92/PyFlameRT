#include <gtest/gtest.h>
#include "pyflame_rt/custom/custom_op.hpp"
#include "pyflame_rt/tensor.hpp"

using namespace pyflame_rt;
using namespace pyflame_rt::custom;

// ============================================================================
// OpSchema Tests
// ============================================================================

TEST(OpSchemaTest, DefaultConstruction) {
    OpSchema schema;
    EXPECT_TRUE(schema.name.empty());
    EXPECT_TRUE(schema.domain.empty());
    EXPECT_EQ(schema.version, 1);
    EXPECT_TRUE(schema.inputs.empty());
    EXPECT_TRUE(schema.outputs.empty());
    EXPECT_TRUE(schema.attributes.empty());
}

TEST(OpSchemaTest, BasicSchema) {
    OpSchema schema;
    schema.name = "MyOp";
    schema.domain = "custom";
    schema.version = 1;
    schema.doc = "My custom operator";

    OpInput input;
    input.name = "X";
    input.dtype = DType::Float32;
    schema.inputs.push_back(input);

    OpOutput output;
    output.name = "Y";
    output.dtype = DType::Float32;
    schema.outputs.push_back(output);

    EXPECT_EQ(schema.name, "MyOp");
    EXPECT_EQ(schema.inputs.size(), 1);
    EXPECT_EQ(schema.outputs.size(), 1);
}

// ============================================================================
// CustomOpRegistry Tests
// ============================================================================

class CustomOpRegistryTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Clear registry before each test
        CustomOpRegistry::instance().clear();
    }

    void TearDown() override {
        // Clean up after each test
        CustomOpRegistry::instance().clear();
    }
};

TEST_F(CustomOpRegistryTest, Singleton) {
    auto& registry1 = CustomOpRegistry::instance();
    auto& registry2 = CustomOpRegistry::instance();
    EXPECT_EQ(&registry1, &registry2);
}

TEST_F(CustomOpRegistryTest, RegisterOp) {
    OpSchema schema;
    schema.name = "TestOp";
    schema.domain = "test";

    CustomOp& op = CustomOpRegistry::instance().register_op(schema);

    EXPECT_EQ(op.name(), "TestOp");
    EXPECT_EQ(op.domain(), "test");
}

TEST_F(CustomOpRegistryTest, RegisterDuplicateThrows) {
    OpSchema schema;
    schema.name = "DuplicateOp";

    CustomOpRegistry::instance().register_op(schema);

    EXPECT_THROW(
        CustomOpRegistry::instance().register_op(schema),
        std::runtime_error
    );
}

TEST_F(CustomOpRegistryTest, GetOp) {
    OpSchema schema;
    schema.name = "GetableOp";
    CustomOpRegistry::instance().register_op(schema);

    CustomOp* op = CustomOpRegistry::instance().get("GetableOp");
    EXPECT_NE(op, nullptr);
    EXPECT_EQ(op->name(), "GetableOp");
}

TEST_F(CustomOpRegistryTest, GetNonExistent) {
    CustomOp* op = CustomOpRegistry::instance().get("NonExistent");
    EXPECT_EQ(op, nullptr);
}

TEST_F(CustomOpRegistryTest, HasOp) {
    OpSchema schema;
    schema.name = "ExistsOp";
    CustomOpRegistry::instance().register_op(schema);

    EXPECT_TRUE(CustomOpRegistry::instance().has("ExistsOp"));
    EXPECT_FALSE(CustomOpRegistry::instance().has("DoesNotExist"));
}

TEST_F(CustomOpRegistryTest, ListOps) {
    OpSchema schema1;
    schema1.name = "Op1";
    CustomOpRegistry::instance().register_op(schema1);

    OpSchema schema2;
    schema2.name = "Op2";
    CustomOpRegistry::instance().register_op(schema2);

    auto ops = CustomOpRegistry::instance().list();
    EXPECT_EQ(ops.size(), 2);
}

TEST_F(CustomOpRegistryTest, UnregisterOp) {
    OpSchema schema;
    schema.name = "ToRemove";
    CustomOpRegistry::instance().register_op(schema);

    EXPECT_TRUE(CustomOpRegistry::instance().has("ToRemove"));

    CustomOpRegistry::instance().unregister("ToRemove");

    EXPECT_FALSE(CustomOpRegistry::instance().has("ToRemove"));
}

TEST_F(CustomOpRegistryTest, Size) {
    EXPECT_EQ(CustomOpRegistry::instance().size(), 0);

    OpSchema schema;
    schema.name = "Op1";
    CustomOpRegistry::instance().register_op(schema);
    EXPECT_EQ(CustomOpRegistry::instance().size(), 1);

    schema.name = "Op2";
    CustomOpRegistry::instance().register_op(schema);
    EXPECT_EQ(CustomOpRegistry::instance().size(), 2);
}

// ============================================================================
// CustomOpBuilder Tests
// ============================================================================

TEST_F(CustomOpRegistryTest, BuilderBasic) {
    CustomOp& op = CustomOpBuilder("BuilderOp")
        .domain("builder_test")
        .version(2)
        .doc("Test op built with builder")
        .input("X", DType::Float32)
        .output("Y", DType::Float32)
        .build();

    EXPECT_EQ(op.name(), "BuilderOp");
    EXPECT_EQ(op.domain(), "builder_test");
    EXPECT_EQ(op.full_name(), "builder_test::BuilderOp");
}

TEST_F(CustomOpRegistryTest, BuilderWithAttributes) {
    CustomOp& op = CustomOpBuilder("AttrOp")
        .attr_int("axis")
        .attr_float("scale")
        .attr_string("mode")
        .attr_ints("perm")
        .attr_floats("weights")
        .build();

    EXPECT_EQ(op.name(), "AttrOp");
}

TEST_F(CustomOpRegistryTest, BuilderWithKernel) {
    auto kernel_fn = [](const std::vector<Tensor>& inputs,
                        const std::unordered_map<std::string, std::any>& attrs)
                        -> std::vector<Tensor> {
        // Simple identity kernel
        return inputs;
    };

    CustomOp& op = CustomOpBuilder("KernelOp")
        .input("X", DType::Float32)
        .output("Y", DType::Float32)
        .kernel(kernel_fn)
        .build();

    // Test execution
    std::vector<Tensor> inputs;
    inputs.push_back(Tensor({2, 2}, DType::Float32));
    inputs[0].fill(1.0f);

    auto outputs = op.execute(inputs, {});

    EXPECT_EQ(outputs.size(), 1);
    EXPECT_EQ(outputs[0].num_elements(), 4);
}

TEST_F(CustomOpRegistryTest, BuilderVariadicInput) {
    CustomOp& op = CustomOpBuilder("VariadicOp")
        .variadic_input("inputs", DType::Float32)
        .output("Y", DType::Float32)
        .build();

    // Should have variadic input
    const auto& schema = op.schema();
    EXPECT_EQ(schema.inputs.size(), 1);
    EXPECT_TRUE(schema.inputs[0].variadic);
}

// ============================================================================
// CustomOp Execution Tests
// ============================================================================

TEST_F(CustomOpRegistryTest, ExecuteWithAttributes) {
    auto kernel_fn = [](const std::vector<Tensor>& inputs,
                        const std::unordered_map<std::string, std::any>& attrs)
                        -> std::vector<Tensor> {
        int64_t scale = std::any_cast<int64_t>(attrs.at("scale"));

        std::vector<Tensor> outputs;
        Tensor out = inputs[0].clone();
        float* data = out.data_ptr<float>();
        for (size_t i = 0; i < out.num_elements(); ++i) {
            data[i] *= static_cast<float>(scale);
        }
        outputs.push_back(out);
        return outputs;
    };

    CustomOp& op = CustomOpBuilder("ScaleOp")
        .input("X", DType::Float32)
        .output("Y", DType::Float32)
        .attr_int("scale", true)
        .kernel(kernel_fn)
        .build();

    std::vector<Tensor> inputs;
    inputs.push_back(Tensor({4}, DType::Float32));
    inputs[0].fill(2.0f);

    std::unordered_map<std::string, std::any> attrs;
    attrs["scale"] = int64_t(3);

    auto outputs = op.execute(inputs, attrs);

    EXPECT_EQ(outputs.size(), 1);
    const float* data = outputs[0].data_ptr<float>();
    EXPECT_FLOAT_EQ(data[0], 6.0f);  // 2.0 * 3
}

// ============================================================================
// Shape and Type Inference Tests
// ============================================================================

TEST_F(CustomOpRegistryTest, ShapeInference) {
    auto shape_fn = [](const std::vector<Shape>& input_shapes) -> std::vector<Shape> {
        // Simple pass-through shape inference
        return input_shapes;
    };

    CustomOp& op = CustomOpBuilder("ShapeInferOp")
        .input("X", DType::Float32)
        .output("Y", DType::Float32)
        .shape_inference(shape_fn)
        .build();

    std::vector<Shape> input_shapes = {{2, 3, 4}};
    auto output_shapes = op.infer_output_shapes(input_shapes);

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes[0].size(), 3);
    EXPECT_EQ(output_shapes[0][0], 2);
    EXPECT_EQ(output_shapes[0][1], 3);
    EXPECT_EQ(output_shapes[0][2], 4);
}

TEST_F(CustomOpRegistryTest, TypeInference) {
    auto type_fn = [](const std::vector<DType>& input_dtypes) -> std::vector<DType> {
        // Output is always Float32
        return {DType::Float32};
    };

    CustomOp& op = CustomOpBuilder("TypeInferOp")
        .input("X")
        .output("Y")
        .type_inference(type_fn)
        .build();

    std::vector<DType> input_dtypes = {DType::Int32};
    auto output_dtypes = op.infer_output_dtypes(input_dtypes);

    EXPECT_EQ(output_dtypes.size(), 1);
    EXPECT_EQ(output_dtypes[0], DType::Float32);
}

// ============================================================================
// Gradient Function Tests
// ============================================================================

TEST_F(CustomOpRegistryTest, GradientFunction) {
    auto grad_fn = [](const std::vector<Tensor>& inputs,
                      const std::vector<Tensor>& grad_outputs)
                      -> std::vector<Tensor> {
        // Simple gradient pass-through
        return grad_outputs;
    };

    CustomOp& op = CustomOpBuilder("GradOp")
        .input("X", DType::Float32)
        .output("Y", DType::Float32)
        .gradient(grad_fn)
        .build();

    EXPECT_TRUE(op.has_gradient());

    std::vector<Tensor> inputs;
    inputs.push_back(Tensor({4}, DType::Float32));

    std::vector<Tensor> grad_outputs;
    grad_outputs.push_back(Tensor({4}, DType::Float32));
    grad_outputs[0].fill(1.0f);

    auto grad_inputs = op.gradient(inputs, grad_outputs);

    EXPECT_EQ(grad_inputs.size(), 1);
}

TEST_F(CustomOpRegistryTest, NoGradientFunction) {
    CustomOp& op = CustomOpBuilder("NoGradOp")
        .input("X", DType::Float32)
        .output("Y", DType::Float32)
        .build();

    EXPECT_FALSE(op.has_gradient());
}

// ============================================================================
// Backend Support Tests
// ============================================================================

TEST_F(CustomOpRegistryTest, BackendSupport) {
    auto cpu_kernel = [](const std::vector<Tensor>& inputs,
                         const std::unordered_map<std::string, std::any>& attrs)
                         -> std::vector<Tensor> {
        return inputs;
    };

    CustomOp& op = CustomOpBuilder("CPUOnlyOp")
        .input("X", DType::Float32)
        .output("Y", DType::Float32)
        .kernel(cpu_kernel, BackendType::CPU)
        .build();

    EXPECT_TRUE(op.supports_backend(BackendType::CPU));
    // May or may not support other backends depending on implementation
}
