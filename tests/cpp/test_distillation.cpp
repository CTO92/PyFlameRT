#include <gtest/gtest.h>
#include "pyflame_rt/distillation/distillation.hpp"
#include "pyflame_rt/tensor.hpp"

using namespace pyflame_rt;
using namespace pyflame_rt::distillation;

// ============================================================================
// DistillationConfig Tests
// ============================================================================

TEST(DistillationConfigTest, DefaultConstruction) {
    DistillationConfig config;
    EXPECT_FLOAT_EQ(config.temperature, 4.0f);
    EXPECT_FLOAT_EQ(config.alpha, 0.7f);
    EXPECT_EQ(config.loss_type, DistillationLoss::KLDivergence);
    EXPECT_TRUE(config.use_hard_labels);
}

TEST(DistillationConfigTest, SoftLabelConfig) {
    auto config = DistillationConfig::soft_label(6.0f);
    EXPECT_FLOAT_EQ(config.temperature, 6.0f);
    EXPECT_EQ(config.loss_type, DistillationLoss::KLDivergence);
}

TEST(DistillationConfigTest, FeatureDistillationConfig) {
    std::vector<std::string> layers = {"layer1", "layer2"};
    auto config = DistillationConfig::feature_distillation(layers);
    EXPECT_EQ(config.feature_layers.size(), 2);
    EXPECT_EQ(config.feature_layers[0], "layer1");
    EXPECT_EQ(config.feature_layers[1], "layer2");
    EXPECT_EQ(config.loss_type, DistillationLoss::MSE);
}

TEST(DistillationConfigTest, AttentionTransferConfig) {
    std::vector<std::string> layers = {"attn1", "attn2", "attn3"};
    auto config = DistillationConfig::attention_transfer(layers);
    EXPECT_EQ(config.attention_layers.size(), 3);
    EXPECT_EQ(config.loss_type, DistillationLoss::Attention);
}

// ============================================================================
// StudentConfig Tests
// ============================================================================

TEST(StudentConfigTest, DefaultConstruction) {
    StudentConfig config;
    EXPECT_FLOAT_EQ(config.hidden_dim_ratio, 1.0f);
    EXPECT_FLOAT_EQ(config.num_layers_ratio, 1.0f);
    EXPECT_FLOAT_EQ(config.num_heads_ratio, 1.0f);
}

TEST(StudentConfigTest, HalfSize) {
    auto config = StudentConfig::half_size();
    EXPECT_FLOAT_EQ(config.hidden_dim_ratio, 0.5f);
    EXPECT_FLOAT_EQ(config.num_layers_ratio, 0.5f);
}

TEST(StudentConfigTest, QuarterSize) {
    auto config = StudentConfig::quarter_size();
    EXPECT_FLOAT_EQ(config.hidden_dim_ratio, 0.25f);
    EXPECT_FLOAT_EQ(config.num_layers_ratio, 0.25f);
}

// ============================================================================
// TrainingConfig Tests
// ============================================================================

TEST(TrainingConfigTest, DefaultConstruction) {
    TrainingConfig config;
    EXPECT_FLOAT_EQ(config.learning_rate, 1e-4f);
    EXPECT_EQ(config.batch_size, 32);
    EXPECT_EQ(config.num_epochs, 10);
    EXPECT_GT(config.gradient_clip, 0.0f);
}

// ============================================================================
// InMemoryDataset Tests
// ============================================================================

TEST(InMemoryDatasetTest, DefaultConstruction) {
    InMemoryDataset dataset;
    EXPECT_EQ(dataset.size(), 0);
}

TEST(InMemoryDatasetTest, AddSample) {
    InMemoryDataset dataset;

    std::unordered_map<std::string, Tensor> inputs;
    inputs["x"] = Tensor({1, 10}, DType::Float32);

    dataset.add_sample(inputs);
    EXPECT_EQ(dataset.size(), 1);

    dataset.add_sample(inputs);
    EXPECT_EQ(dataset.size(), 2);
}

TEST(InMemoryDatasetTest, GetBatch) {
    InMemoryDataset dataset;

    // Add 10 samples
    for (int i = 0; i < 10; ++i) {
        std::unordered_map<std::string, Tensor> inputs;
        inputs["x"] = Tensor({1, 10}, DType::Float32);
        inputs["x"].fill(static_cast<float>(i));
        dataset.add_sample(inputs);
    }

    EXPECT_EQ(dataset.size(), 10);

    // Get batch of 3
    auto batch = dataset.get_batch(0, 3);
    EXPECT_EQ(batch.size(), 3);
}

TEST(InMemoryDatasetTest, Clear) {
    InMemoryDataset dataset;

    std::unordered_map<std::string, Tensor> inputs;
    inputs["x"] = Tensor({1, 10}, DType::Float32);
    dataset.add_sample(inputs);
    dataset.add_sample(inputs);

    EXPECT_EQ(dataset.size(), 2);

    dataset.clear();
    EXPECT_EQ(dataset.size(), 0);
}

// ============================================================================
// DistillationResult Tests
// ============================================================================

TEST(DistillationResultTest, CompressionRatio) {
    DistillationResult result;
    result.teacher_size_bytes = 1000000;  // 1MB
    result.student_size_bytes = 250000;   // 250KB

    EXPECT_FLOAT_EQ(result.compression_ratio(), 4.0f);
}

// ============================================================================
// Loss Function Tests
// ============================================================================

TEST(DistillationLossTest, ComputeSoftTargets) {
    Tensor logits({1, 4}, DType::Float32);
    float* data = logits.data_ptr<float>();
    data[0] = 1.0f;
    data[1] = 2.0f;
    data[2] = 3.0f;
    data[3] = 4.0f;

    Tensor soft_targets = compute_soft_targets(logits, 2.0f);

    // Soft targets should sum to 1
    const float* soft = soft_targets.data_ptr<float>();
    float sum = 0.0f;
    for (int i = 0; i < 4; ++i) {
        sum += soft[i];
        EXPECT_GT(soft[i], 0.0f);  // All positive
        EXPECT_LT(soft[i], 1.0f);  // All less than 1
    }
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

TEST(DistillationLossTest, SoftTargetsWithHighTemperature) {
    Tensor logits({1, 4}, DType::Float32);
    float* data = logits.data_ptr<float>();
    data[0] = 0.0f;
    data[1] = 0.0f;
    data[2] = 0.0f;
    data[3] = 10.0f;  // One dominant class

    // With low temperature, should be close to one-hot
    Tensor soft_low = compute_soft_targets(logits, 0.1f);

    // With high temperature, should be more uniform
    Tensor soft_high = compute_soft_targets(logits, 10.0f);

    const float* low = soft_low.data_ptr<float>();
    const float* high = soft_high.data_ptr<float>();

    // High temperature should be more uniform
    float variance_low = 0.0f, variance_high = 0.0f;
    float mean = 0.25f;  // Expected mean for 4 classes

    for (int i = 0; i < 4; ++i) {
        variance_low += (low[i] - mean) * (low[i] - mean);
        variance_high += (high[i] - mean) * (high[i] - mean);
    }

    // High temperature should have lower variance (more uniform)
    EXPECT_LT(variance_high, variance_low);
}

TEST(DistillationLossTest, KLDivergenceLoss) {
    Tensor student_logits({1, 4}, DType::Float32);
    float* s_data = student_logits.data_ptr<float>();
    s_data[0] = 1.0f;
    s_data[1] = 2.0f;
    s_data[2] = 1.0f;
    s_data[3] = 1.0f;

    Tensor teacher_logits({1, 4}, DType::Float32);
    float* t_data = teacher_logits.data_ptr<float>();
    t_data[0] = 1.0f;
    t_data[1] = 3.0f;  // Slightly different
    t_data[2] = 1.0f;
    t_data[3] = 1.0f;

    float loss = kl_divergence_loss(student_logits, teacher_logits, 4.0f);

    // KL divergence should be non-negative
    EXPECT_GE(loss, 0.0f);

    // Same distributions should have zero loss
    float same_loss = kl_divergence_loss(student_logits, student_logits, 4.0f);
    EXPECT_NEAR(same_loss, 0.0f, 1e-5f);
}

TEST(DistillationLossTest, FeatureLossMSE) {
    Tensor student_features({1, 10}, DType::Float32);
    student_features.fill(1.0f);

    Tensor teacher_features({1, 10}, DType::Float32);
    teacher_features.fill(2.0f);

    float loss = feature_loss(student_features, teacher_features, DistillationLoss::MSE);

    // MSE of (1-2)^2 = 1.0 for each element
    EXPECT_NEAR(loss, 1.0f, 1e-5f);

    // Same features should have zero loss
    float same_loss = feature_loss(student_features, student_features, DistillationLoss::MSE);
    EXPECT_NEAR(same_loss, 0.0f, 1e-5f);
}

TEST(DistillationLossTest, FeatureLossCosine) {
    // Orthogonal vectors
    Tensor a({1, 4}, DType::Float32);
    float* a_data = a.data_ptr<float>();
    a_data[0] = 1.0f;
    a_data[1] = 0.0f;
    a_data[2] = 0.0f;
    a_data[3] = 0.0f;

    Tensor b({1, 4}, DType::Float32);
    float* b_data = b.data_ptr<float>();
    b_data[0] = 0.0f;
    b_data[1] = 1.0f;
    b_data[2] = 0.0f;
    b_data[3] = 0.0f;

    float orthogonal_loss = feature_loss(a, b, DistillationLoss::Cosine);

    // Same vector should have zero loss
    float same_loss = feature_loss(a, a, DistillationLoss::Cosine);
    EXPECT_NEAR(same_loss, 0.0f, 1e-5f);

    // Orthogonal vectors should have higher loss
    EXPECT_GT(orthogonal_loss, same_loss);
}
