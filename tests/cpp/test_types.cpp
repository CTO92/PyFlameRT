#include <gtest/gtest.h>
#include "pyflame_rt/types.hpp"

using namespace pyflame_rt;

TEST(DTypeTest, Size) {
    EXPECT_EQ(dtype_size(DType::Float32), 4);
    EXPECT_EQ(dtype_size(DType::Float16), 2);
    EXPECT_EQ(dtype_size(DType::Float64), 8);
    EXPECT_EQ(dtype_size(DType::Int64), 8);
    EXPECT_EQ(dtype_size(DType::Int32), 4);
    EXPECT_EQ(dtype_size(DType::Int8), 1);
    EXPECT_EQ(dtype_size(DType::Bool), 1);
}

TEST(DTypeTest, Name) {
    EXPECT_EQ(dtype_name(DType::Float32), "float32");
    EXPECT_EQ(dtype_name(DType::Float16), "float16");
    EXPECT_EQ(dtype_name(DType::Int64), "int64");
}

TEST(DTypeTest, FromName) {
    EXPECT_EQ(dtype_from_name("float32"), DType::Float32);
    EXPECT_EQ(dtype_from_name("int64"), DType::Int64);
    EXPECT_THROW(dtype_from_name("invalid"), std::invalid_argument);
}

TEST(ShapeTest, ToString) {
    Shape shape = {1, 3, std::nullopt, 224};
    EXPECT_EQ(shape_to_string(shape), "[1, 3, ?, 224]");
}

TEST(ShapeTest, IsDynamic) {
    Shape static_shape = {1, 3, 224, 224};
    Shape dynamic_shape = {std::nullopt, 3, 224, 224};

    EXPECT_FALSE(is_dynamic_shape(static_shape));
    EXPECT_TRUE(is_dynamic_shape(dynamic_shape));
}

TEST(ShapeTest, NumElements) {
    Shape shape = {2, 3, 4};
    auto num = shape_num_elements(shape);
    EXPECT_TRUE(num.has_value());
    EXPECT_EQ(num.value(), 24);

    Shape dynamic_shape = {2, std::nullopt, 4};
    EXPECT_FALSE(shape_num_elements(dynamic_shape).has_value());
}

TEST(TensorInfoTest, Basic) {
    TensorInfo info("input", {{1}, {3}, {224}, {224}}, DType::Float32);

    EXPECT_EQ(info.name, "input");
    EXPECT_EQ(info.dtype, DType::Float32);
    EXPECT_FALSE(info.is_dynamic());

    auto num_elem = info.num_elements();
    EXPECT_TRUE(num_elem.has_value());
    EXPECT_EQ(num_elem.value(), 1 * 3 * 224 * 224);

    auto size = info.size_bytes();
    EXPECT_TRUE(size.has_value());
    EXPECT_EQ(size.value(), 1 * 3 * 224 * 224 * 4);
}

TEST(NodeArgTest, FromTensorInfo) {
    TensorInfo info("output", {{1}, {1000}}, DType::Float32);
    NodeArg arg = NodeArg::from_tensor_info(info);

    EXPECT_EQ(arg.name, "output");
    EXPECT_EQ(arg.type_str, "tensor(float32)");
    EXPECT_EQ(arg.shape.size(), 2);
    EXPECT_EQ(arg.shape[0].value(), 1);
    EXPECT_EQ(arg.shape[1].value(), 1000);
}

TEST(ModelMetadataTest, Default) {
    ModelMetadata meta;
    EXPECT_EQ(meta.producer_name, "");
    EXPECT_EQ(meta.version, 0);
    EXPECT_TRUE(meta.custom_metadata.empty());
}
