#include <gtest/gtest.h>
#include "pyflame_rt/tensor.hpp"

using namespace pyflame_rt;

TEST(TensorTest, DefaultConstruction) {
    Tensor t;
    EXPECT_FALSE(t.is_valid());
    EXPECT_FALSE(t.owns_data());
}

TEST(TensorTest, Construction) {
    Tensor t({2, 3, 4}, DType::Float32);
    EXPECT_TRUE(t.is_valid());
    EXPECT_TRUE(t.owns_data());
    EXPECT_EQ(t.ndim(), 3);
    EXPECT_EQ(t.shape()[0], 2);
    EXPECT_EQ(t.shape()[1], 3);
    EXPECT_EQ(t.shape()[2], 4);
    EXPECT_EQ(t.num_elements(), 24);
    EXPECT_EQ(t.size_bytes(), 24 * 4);
}

TEST(TensorTest, DifferentDTypes) {
    Tensor f32({10}, DType::Float32);
    EXPECT_EQ(f32.size_bytes(), 10 * 4);

    Tensor i64({10}, DType::Int64);
    EXPECT_EQ(i64.size_bytes(), 10 * 8);

    Tensor f16({10}, DType::Float16);
    EXPECT_EQ(f16.size_bytes(), 10 * 2);
}

TEST(TensorTest, Fill) {
    Tensor t({2, 2}, DType::Float32);
    t.fill(3.14f);

    const float* data = t.data_ptr<float>();
    EXPECT_FLOAT_EQ(data[0], 3.14f);
    EXPECT_FLOAT_EQ(data[1], 3.14f);
    EXPECT_FLOAT_EQ(data[2], 3.14f);
    EXPECT_FLOAT_EQ(data[3], 3.14f);
}

TEST(TensorTest, Zero) {
    Tensor t({4}, DType::Float32);
    t.fill(1.0f);
    t.zero();

    const float* data = t.data_ptr<float>();
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(data[i], 0.0f);
    }
}

TEST(TensorTest, Clone) {
    Tensor t({2, 2}, DType::Float32);
    t.fill(1.0f);

    Tensor clone = t.clone();

    // Verify clone has same data
    EXPECT_FLOAT_EQ(clone.data_ptr<float>()[0], 1.0f);

    // Modify original
    t.fill(2.0f);

    // Clone should be unchanged
    EXPECT_FLOAT_EQ(clone.data_ptr<float>()[0], 1.0f);
    EXPECT_FLOAT_EQ(t.data_ptr<float>()[0], 2.0f);
}

TEST(TensorTest, View) {
    Tensor t({2, 2}, DType::Float32);
    t.fill(5.0f);

    Tensor view = t.view();

    // View should not own data
    EXPECT_FALSE(view.owns_data());
    EXPECT_TRUE(t.owns_data());

    // Same underlying data
    EXPECT_EQ(view.data(), t.data());
}

TEST(TensorTest, MoveConstruction) {
    Tensor t1({2, 2}, DType::Float32);
    void* original_data = t1.data();

    Tensor t2 = std::move(t1);

    EXPECT_EQ(t2.data(), original_data);
    EXPECT_FALSE(t1.is_valid());
    EXPECT_TRUE(t2.is_valid());
}

TEST(TensorTest, MoveAssignment) {
    Tensor t1({2, 2}, DType::Float32);
    void* original_data = t1.data();

    Tensor t2;
    t2 = std::move(t1);

    EXPECT_EQ(t2.data(), original_data);
    EXPECT_FALSE(t1.is_valid());
    EXPECT_TRUE(t2.is_valid());
}

TEST(TensorTest, CopyConstruction) {
    Tensor t1({2, 2}, DType::Float32);
    t1.fill(42.0f);

    Tensor t2(t1);

    // Different memory
    EXPECT_NE(t1.data(), t2.data());
    // Same values
    EXPECT_FLOAT_EQ(t2.data_ptr<float>()[0], 42.0f);
}

TEST(TensorTest, Reshape) {
    Tensor t({2, 3}, DType::Float32);
    t.fill(1.0f);

    Tensor reshaped = t.reshape({3, 2});

    EXPECT_EQ(reshaped.shape()[0], 3);
    EXPECT_EQ(reshaped.shape()[1], 2);
    EXPECT_EQ(reshaped.num_elements(), 6);
}

TEST(TensorTest, ReshapeInvalidSize) {
    Tensor t({2, 3}, DType::Float32);

    EXPECT_THROW(t.reshape({2, 2}), std::invalid_argument);
}

TEST(TensorTest, EmptyTensor) {
    Tensor t({0}, DType::Float32);
    EXPECT_EQ(t.num_elements(), 0);
    EXPECT_EQ(t.size_bytes(), 0);
}
