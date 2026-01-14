#include <gtest/gtest.h>
#include "pyflame_rt/pruning/pruning.hpp"
#include "pyflame_rt/pruning/sparse_tensor.hpp"
#include "pyflame_rt/graph.hpp"
#include "pyflame_rt/tensor.hpp"

using namespace pyflame_rt;
using namespace pyflame_rt::pruning;

// ============================================================================
// PruningConfig Tests
// ============================================================================

TEST(PruningConfigTest, DefaultConstruction) {
    PruningConfig config;
    EXPECT_FLOAT_EQ(config.target_sparsity, 0.5f);
    EXPECT_EQ(config.granularity, PruningGranularity::Unstructured);
    EXPECT_EQ(config.criterion, PruningCriterion::Magnitude);
    EXPECT_EQ(config.schedule, PruningSchedule::OneShot);
}

TEST(PruningConfigTest, MagnitudePruning) {
    auto config = PruningConfig::magnitude_pruning(0.7f);
    EXPECT_FLOAT_EQ(config.target_sparsity, 0.7f);
    EXPECT_EQ(config.granularity, PruningGranularity::Unstructured);
    EXPECT_EQ(config.criterion, PruningCriterion::Magnitude);
}

TEST(PruningConfigTest, StructuredPruning) {
    auto config = PruningConfig::structured_pruning(0.5f);
    EXPECT_FLOAT_EQ(config.target_sparsity, 0.5f);
    EXPECT_EQ(config.granularity, PruningGranularity::Structured);
}

TEST(PruningConfigTest, NMSparsity) {
    auto config = PruningConfig::nm_sparsity(2, 4);
    EXPECT_EQ(config.n_value, 2);
    EXPECT_EQ(config.m_value, 4);
    EXPECT_EQ(config.granularity, PruningGranularity::NM);
    EXPECT_FLOAT_EQ(config.target_sparsity, 0.5f);  // 2:4 = 50% sparsity
}

// ============================================================================
// PruningMask Tests
// ============================================================================

TEST(PruningMaskTest, DefaultConstruction) {
    PruningMask mask;
    EXPECT_TRUE(mask.shape().empty());
}

TEST(PruningMaskTest, FromTensor) {
    // Create a tensor with some zero values
    Tensor t({4}, DType::Float32);
    float* data = t.data_ptr<float>();
    data[0] = 1.0f;
    data[1] = 0.0f;
    data[2] = 2.0f;
    data[3] = 0.0f;

    PruningMask mask = PruningMask::from_tensor(t);

    EXPECT_EQ(mask.shape().size(), 1);
    EXPECT_EQ(mask.shape()[0], 4);
    EXPECT_EQ(mask.num_zeros(), 2);
    EXPECT_EQ(mask.num_nonzeros(), 2);
    EXPECT_FLOAT_EQ(mask.sparsity(), 0.5f);
}

TEST(PruningMaskTest, ApplyMask) {
    // Create tensor
    Tensor t({4}, DType::Float32);
    float* data = t.data_ptr<float>();
    data[0] = 1.0f;
    data[1] = 2.0f;
    data[2] = 3.0f;
    data[3] = 4.0f;

    // Create mask tensor with 50% sparsity
    Tensor mask_tensor({4}, DType::Float32);
    float* mask_data = mask_tensor.data_ptr<float>();
    mask_data[0] = 1.0f;
    mask_data[1] = 0.0f;
    mask_data[2] = 1.0f;
    mask_data[3] = 0.0f;

    PruningMask mask = PruningMask::from_tensor(mask_tensor);
    Tensor result = mask.apply(t);

    const float* result_data = result.data_ptr<float>();
    EXPECT_FLOAT_EQ(result_data[0], 1.0f);
    EXPECT_FLOAT_EQ(result_data[1], 0.0f);
    EXPECT_FLOAT_EQ(result_data[2], 3.0f);
    EXPECT_FLOAT_EQ(result_data[3], 0.0f);
}

// ============================================================================
// PruningStats Tests
// ============================================================================

TEST(PruningStatsTest, CompressionRatio) {
    PruningStats stats;
    stats.original_size_bytes = 1000;
    stats.pruned_size_bytes = 500;

    EXPECT_FLOAT_EQ(stats.compression_ratio(), 2.0f);
}

TEST(PruningStatsTest, ActualSparsity) {
    PruningStats stats;
    stats.total_params = 100;
    stats.pruned_params = 75;

    EXPECT_FLOAT_EQ(stats.actual_sparsity(), 0.75f);
}

// ============================================================================
// SparseTensor Tests
// ============================================================================

TEST(SparseTensorTest, DefaultConstruction) {
    SparseTensor sparse;
    EXPECT_TRUE(sparse.shape().empty());
}

TEST(SparseTensorTest, FromDense) {
    // Create a tensor with some zero values
    Tensor dense({2, 3}, DType::Float32);
    float* data = dense.data_ptr<float>();
    data[0] = 1.0f;
    data[1] = 0.0f;
    data[2] = 2.0f;
    data[3] = 0.0f;
    data[4] = 0.0f;
    data[5] = 3.0f;

    SparseTensor sparse = SparseTensor::from_dense(dense, SparseFormat::COO);

    EXPECT_EQ(sparse.format(), SparseFormat::COO);
    EXPECT_EQ(sparse.shape().size(), 2);
    EXPECT_EQ(sparse.shape()[0], 2);
    EXPECT_EQ(sparse.shape()[1], 3);
    EXPECT_EQ(sparse.nnz(), 3);  // 3 non-zero values
    EXPECT_FLOAT_EQ(sparse.sparsity(), 0.5f);  // 3/6 = 50%
}

TEST(SparseTensorTest, ToDense) {
    // Create dense, convert to sparse, convert back
    Tensor original({2, 2}, DType::Float32);
    float* data = original.data_ptr<float>();
    data[0] = 1.0f;
    data[1] = 0.0f;
    data[2] = 0.0f;
    data[3] = 2.0f;

    SparseTensor sparse = SparseTensor::from_dense(original, SparseFormat::COO);
    Tensor reconstructed = sparse.to_dense();

    const float* result = reconstructed.data_ptr<float>();
    EXPECT_FLOAT_EQ(result[0], 1.0f);
    EXPECT_FLOAT_EQ(result[1], 0.0f);
    EXPECT_FLOAT_EQ(result[2], 0.0f);
    EXPECT_FLOAT_EQ(result[3], 2.0f);
}

TEST(SparseTensorTest, CompressionRatio) {
    Tensor dense({100}, DType::Float32);
    dense.zero();

    // Set 10% of values to non-zero
    float* data = dense.data_ptr<float>();
    for (int i = 0; i < 10; ++i) {
        data[i * 10] = 1.0f;
    }

    SparseTensor sparse = SparseTensor::from_dense(dense, SparseFormat::COO);

    // Sparse should use less memory than dense
    EXPECT_LT(sparse.memory_bytes(), dense.size_bytes());
    EXPECT_GT(sparse.compression_ratio(), 1.0f);
}

TEST(SparseTensorTest, FormatConversion) {
    Tensor dense({3, 3}, DType::Float32);
    float* data = dense.data_ptr<float>();
    // Diagonal matrix
    for (int i = 0; i < 9; ++i) {
        data[i] = (i % 4 == 0) ? 1.0f : 0.0f;
    }

    SparseTensor coo = SparseTensor::from_dense(dense, SparseFormat::COO);
    SparseTensor csr = coo.to_csr();
    SparseTensor csc = coo.to_csc();

    EXPECT_EQ(csr.format(), SparseFormat::CSR);
    EXPECT_EQ(csc.format(), SparseFormat::CSC);
    EXPECT_EQ(csr.nnz(), coo.nnz());
    EXPECT_EQ(csc.nnz(), coo.nnz());
}

// ============================================================================
// WeightPruner Tests
// ============================================================================

TEST(WeightPrunerTest, GetSparsityAtStep) {
    PruningConfig config;
    config.schedule = PruningSchedule::Iterative;
    config.target_sparsity = 0.9f;
    config.initial_sparsity = 0.0f;
    config.start_step = 0;
    config.end_step = 100;

    WeightPruner pruner(config);

    // At step 0, should be initial sparsity
    EXPECT_FLOAT_EQ(pruner.get_sparsity_at_step(0), 0.0f);

    // At step 50, should be halfway
    float mid_sparsity = pruner.get_sparsity_at_step(50);
    EXPECT_GT(mid_sparsity, 0.0f);
    EXPECT_LT(mid_sparsity, 0.9f);

    // At step 100+, should be target sparsity
    EXPECT_FLOAT_EQ(pruner.get_sparsity_at_step(100), 0.9f);
    EXPECT_FLOAT_EQ(pruner.get_sparsity_at_step(200), 0.9f);
}

TEST(WeightPrunerTest, OneShotSchedule) {
    PruningConfig config;
    config.schedule = PruningSchedule::OneShot;
    config.target_sparsity = 0.5f;

    WeightPruner pruner(config);

    // One-shot should always return target sparsity
    EXPECT_FLOAT_EQ(pruner.get_sparsity_at_step(0), 0.5f);
    EXPECT_FLOAT_EQ(pruner.get_sparsity_at_step(50), 0.5f);
    EXPECT_FLOAT_EQ(pruner.get_sparsity_at_step(100), 0.5f);
}
