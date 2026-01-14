#include <gtest/gtest.h>
#include "pyflame_rt/registry.hpp"

using namespace pyflame_rt;

class RegistryTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Clear registry before each test
        OperatorRegistry::instance().clear();
    }

    void TearDown() override {
        OperatorRegistry::instance().clear();
    }
};

TEST_F(RegistryTest, RegisterAndGet) {
    auto& registry = OperatorRegistry::instance();

    OpFunc add_func = [](const std::vector<const Tensor*>& inputs,
                         const OpContext& ctx) -> std::vector<Tensor> {
        return {inputs[0]->clone()};
    };

    registry.register_op("TestAdd", add_func);

    EXPECT_TRUE(registry.has("TestAdd"));
    EXPECT_FALSE(registry.has("NonExistent"));

    const OpFunc* retrieved = registry.get("TestAdd");
    EXPECT_NE(retrieved, nullptr);
}

TEST_F(RegistryTest, ListOps) {
    auto& registry = OperatorRegistry::instance();

    registry.register_op("Op1", [](const std::vector<const Tensor*>&,
                                   const OpContext&) { return std::vector<Tensor>{}; });
    registry.register_op("Op2", [](const std::vector<const Tensor*>&,
                                   const OpContext&) { return std::vector<Tensor>{}; });
    registry.register_op("Op3", [](const std::vector<const Tensor*>&,
                                   const OpContext&) { return std::vector<Tensor>{}; });

    auto ops = registry.list_ops();

    EXPECT_EQ(ops.size(), 3);
    // Should be sorted
    EXPECT_EQ(ops[0], "Op1");
    EXPECT_EQ(ops[1], "Op2");
    EXPECT_EQ(ops[2], "Op3");
}

TEST_F(RegistryTest, Unregister) {
    auto& registry = OperatorRegistry::instance();

    registry.register_op("ToRemove", [](const std::vector<const Tensor*>&,
                                        const OpContext&) { return std::vector<Tensor>{}; });

    EXPECT_TRUE(registry.has("ToRemove"));
    EXPECT_TRUE(registry.unregister_op("ToRemove"));
    EXPECT_FALSE(registry.has("ToRemove"));
    EXPECT_FALSE(registry.unregister_op("ToRemove"));  // Already removed
}

TEST_F(RegistryTest, ExecuteOp) {
    auto& registry = OperatorRegistry::instance();

    // Register a simple identity operator
    registry.register_op("Identity", [](const std::vector<const Tensor*>& inputs,
                                        const OpContext& ctx) -> std::vector<Tensor> {
        return {inputs[0]->clone()};
    });

    Tensor input({2, 2}, DType::Float32);
    input.fill(42.0f);

    Node node("id_0", "Identity", {"x"}, {"y"});
    OpContext ctx{&node};

    const OpFunc* func = registry.get("Identity");
    ASSERT_NE(func, nullptr);

    std::vector<const Tensor*> inputs = {&input};
    auto outputs = (*func)(inputs, ctx);

    EXPECT_EQ(outputs.size(), 1);
    EXPECT_FLOAT_EQ(outputs[0].data_ptr<float>()[0], 42.0f);
}
