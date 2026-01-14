#pragma once

#include "pyflame_rt/tensor.hpp"
#include "pyflame_rt/graph.hpp"
#include "pyflame_rt/session.hpp"
#include <memory>
#include <vector>
#include <functional>
#include <optional>
#include <unordered_map>

namespace pyflame_rt {
namespace distillation {

/// Distillation loss type
enum class DistillationLoss {
    KLDivergence,    // KL divergence for soft targets
    MSE,             // Mean squared error for features
    Cosine,          // Cosine similarity loss
    Attention,       // Attention transfer
    Hint             // Intermediate layer hints
};

/// Layer matching strategy for feature distillation
enum class LayerMatching {
    Manual,          // User-specified layer mapping
    Automatic,       // Automatic matching by shape
    Sequential       // Match by layer index
};

/// Configuration for knowledge distillation
struct DistillationConfig {
    /// Temperature for softmax (higher = softer)
    float temperature = 4.0f;

    /// Alpha weight for distillation loss (vs hard label loss)
    float alpha = 0.7f;

    /// Distillation loss type
    DistillationLoss loss_type = DistillationLoss::KLDivergence;

    /// Layer matching strategy
    LayerMatching matching = LayerMatching::Automatic;

    /// Manual layer mapping (teacher -> student)
    std::vector<std::pair<std::string, std::string>> layer_mapping;

    /// Layers to distill (empty = output only)
    std::vector<std::string> teacher_layers;

    /// Corresponding student layers
    std::vector<std::string> student_layers;

    /// Use intermediate feature distillation
    bool use_feature_distillation = false;

    /// Feature loss weight
    float feature_loss_weight = 0.1f;

    /// Use attention transfer
    bool use_attention_transfer = false;

    /// Attention loss weight
    float attention_loss_weight = 0.05f;

    /// Training epochs
    size_t epochs = 10;

    /// Batch size
    size_t batch_size = 32;

    /// Learning rate
    float learning_rate = 1e-4f;

    /// Learning rate decay
    float lr_decay = 0.1f;

    /// Epochs before LR decay
    std::vector<size_t> lr_decay_epochs = {5, 8};

    /// Weight decay
    float weight_decay = 1e-5f;

    /// Early stopping patience (0 = disabled)
    size_t early_stopping_patience = 0;

    /// Validation frequency (epochs)
    size_t validation_frequency = 1;

    /// Output name for logits
    std::string output_name = "output";
};

/// Distillation training state
struct DistillationState {
    size_t current_epoch = 0;
    size_t total_steps = 0;
    float current_lr = 0.0f;
    float best_loss = std::numeric_limits<float>::max();
    size_t steps_without_improvement = 0;
};

/// Distillation metrics
struct DistillationMetrics {
    /// Total loss (combined)
    float total_loss = 0.0f;

    /// Hard label loss
    float hard_loss = 0.0f;

    /// Soft (distillation) loss
    float soft_loss = 0.0f;

    /// Feature distillation loss
    float feature_loss = 0.0f;

    /// Attention transfer loss
    float attention_loss = 0.0f;

    /// Student accuracy
    float student_accuracy = 0.0f;

    /// Teacher accuracy (for reference)
    float teacher_accuracy = 0.0f;

    /// Number of samples processed
    size_t num_samples = 0;
};

/// Data batch for distillation
struct DistillationBatch {
    /// Input tensors
    std::unordered_map<std::string, Tensor> inputs;

    /// Ground truth labels (optional)
    std::optional<Tensor> labels;

    /// Batch size
    size_t batch_size = 0;
};

/// Dataset interface for distillation
class DistillationDataset {
public:
    virtual ~DistillationDataset() = default;

    /// Get number of samples
    virtual size_t size() const = 0;

    /// Get a batch of data
    virtual DistillationBatch get_batch(size_t start, size_t batch_size) = 0;

    /// Shuffle dataset
    virtual void shuffle() = 0;

    /// Reset to beginning
    virtual void reset() = 0;
};

/// In-memory dataset implementation
class InMemoryDataset : public DistillationDataset {
public:
    InMemoryDataset() = default;

    InMemoryDataset(
        const std::vector<std::unordered_map<std::string, Tensor>>& inputs,
        const std::vector<Tensor>& labels = {});

    /// Add a sample
    void add_sample(const std::unordered_map<std::string, Tensor>& inputs,
                    const Tensor& label = Tensor());

    /// Add samples from tensors
    void add_samples(const Tensor& inputs, const Tensor& labels,
                     const std::string& input_name = "input");

    size_t size() const override;
    DistillationBatch get_batch(size_t start, size_t batch_size) override;
    void shuffle() override;
    void reset() override;

private:
    std::vector<std::unordered_map<std::string, Tensor>> inputs_;
    std::vector<Tensor> labels_;
    std::vector<size_t> indices_;
    bool has_labels_ = false;
};

/// Teacher-student pair
struct TeacherStudent {
    std::shared_ptr<InferenceSession> teacher;
    std::shared_ptr<InferenceSession> student;
    Graph student_graph;  // For weight updates
};

/// Distillation trainer
class DistillationTrainer {
public:
    explicit DistillationTrainer(const DistillationConfig& config);
    ~DistillationTrainer();

    // ========================================================================
    // Setup
    // ========================================================================

    /// Set teacher model from session
    void set_teacher(std::shared_ptr<InferenceSession> teacher);

    /// Set teacher model from path
    void set_teacher(const std::string& model_path);

    /// Set student model from session
    void set_student(std::shared_ptr<InferenceSession> student,
                     const Graph& student_graph);

    /// Set student model from path
    void set_student(const std::string& model_path);

    /// Set training dataset
    void set_train_dataset(std::shared_ptr<DistillationDataset> dataset);

    /// Set validation dataset
    void set_validation_dataset(std::shared_ptr<DistillationDataset> dataset);

    // ========================================================================
    // Training
    // ========================================================================

    /// Training callback type
    using TrainingCallback = std::function<void(
        size_t epoch,
        const DistillationMetrics& train_metrics,
        const DistillationMetrics& val_metrics,
        const DistillationState& state)>;

    /// Run distillation training
    Graph train(TrainingCallback callback = nullptr);

    /// Run single epoch
    DistillationMetrics train_epoch();

    /// Run validation
    DistillationMetrics validate();

    /// Process a single batch
    DistillationMetrics process_batch(const DistillationBatch& batch, bool training = true);

    // ========================================================================
    // Loss Computation
    // ========================================================================

    /// Compute distillation loss
    float compute_distillation_loss(
        const Tensor& teacher_logits,
        const Tensor& student_logits,
        float temperature);

    /// Compute feature loss
    float compute_feature_loss(
        const std::vector<Tensor>& teacher_features,
        const std::vector<Tensor>& student_features);

    /// Compute attention loss
    float compute_attention_loss(
        const std::vector<Tensor>& teacher_attention,
        const std::vector<Tensor>& student_attention);

    /// Compute hard label loss (cross-entropy)
    float compute_hard_loss(
        const Tensor& logits,
        const Tensor& labels);

    // ========================================================================
    // Utilities
    // ========================================================================

    /// Apply softmax with temperature
    Tensor softmax_with_temperature(const Tensor& logits, float temperature);

    /// Compute KL divergence
    Tensor kl_divergence(const Tensor& p, const Tensor& q);

    /// Compute MSE loss
    float mse_loss(const Tensor& a, const Tensor& b);

    /// Compute cosine similarity loss
    float cosine_loss(const Tensor& a, const Tensor& b);

    // ========================================================================
    // State
    // ========================================================================

    /// Get current training state
    const DistillationState& state() const { return state_; }

    /// Get configuration
    const DistillationConfig& config() const { return config_; }

    /// Get best student model
    Graph get_best_model() const { return best_model_; }

    /// Check if training is complete
    bool is_complete() const;

private:
    DistillationConfig config_;
    DistillationState state_;
    TeacherStudent models_;
    std::shared_ptr<DistillationDataset> train_dataset_;
    std::shared_ptr<DistillationDataset> val_dataset_;
    Graph best_model_;

    void update_learning_rate();
    bool check_early_stopping(float loss);
    void save_best_model();
};

/// Create a student model from teacher (architecture compression)
struct StudentArchitectureConfig {
    /// Compression ratio
    float compression_ratio = 0.5f;

    /// Minimum layer width
    size_t min_width = 32;

    /// Preserve first/last layers
    bool preserve_endpoints = true;

    /// Use uniform compression
    bool uniform_compression = false;

    /// Number of layers to reduce (for depth reduction)
    size_t reduce_layers = 0;
};

/// Create compressed student architecture from teacher
Graph create_student_architecture(
    const Graph& teacher,
    const StudentArchitectureConfig& config);

/// Convenience functions

/// Distill a model using default settings
Graph distill_model(
    const Graph& teacher,
    const Graph& student,
    std::shared_ptr<DistillationDataset> dataset,
    const DistillationConfig& config = {});

/// Distill from model paths
Graph distill_model(
    const std::string& teacher_path,
    const std::string& student_path,
    std::shared_ptr<DistillationDataset> dataset,
    const DistillationConfig& config = {});

/// Auto-create student and distill
Graph auto_distill(
    const Graph& teacher,
    std::shared_ptr<DistillationDataset> dataset,
    float compression_ratio = 0.5f,
    const DistillationConfig& config = {});

} // namespace distillation
} // namespace pyflame_rt
