#include "pyflame_rt/distillation/distillation.hpp"
#include <algorithm>
#include <stdexcept>

namespace pyflame_rt {
namespace distillation {

DistillationTrainer::DistillationTrainer(const DistillationConfig& config)
    : config_(config)
{
    state_.current_lr = config_.learning_rate;
}

DistillationTrainer::~DistillationTrainer() = default;

void DistillationTrainer::set_teacher(std::shared_ptr<InferenceSession> teacher) {
    models_.teacher = std::move(teacher);
}

void DistillationTrainer::set_teacher(const std::string& model_path) {
    models_.teacher = std::make_shared<InferenceSession>(model_path);
}

void DistillationTrainer::set_student(std::shared_ptr<InferenceSession> student,
                                       const Graph& student_graph) {
    models_.student = std::move(student);
    models_.student_graph = student_graph;
    best_model_ = student_graph;
}

void DistillationTrainer::set_student(const std::string& model_path) {
    models_.student = std::make_shared<InferenceSession>(model_path);
    // Note: without access to the graph, we can't update weights
    // This is a simplified implementation
}

void DistillationTrainer::set_train_dataset(
    std::shared_ptr<DistillationDataset> dataset) {
    train_dataset_ = std::move(dataset);
}

void DistillationTrainer::set_validation_dataset(
    std::shared_ptr<DistillationDataset> dataset) {
    val_dataset_ = std::move(dataset);
}

Graph DistillationTrainer::train(TrainingCallback callback) {
    if (!models_.teacher || !models_.student) {
        throw std::runtime_error("Teacher and student models must be set before training");
    }

    if (!train_dataset_) {
        throw std::runtime_error("Training dataset must be set before training");
    }

    best_model_ = models_.student_graph;
    state_.best_loss = std::numeric_limits<float>::max();

    for (size_t epoch = 0; epoch < config_.epochs; ++epoch) {
        state_.current_epoch = epoch;

        // Shuffle and reset dataset
        train_dataset_->shuffle();
        train_dataset_->reset();

        // Train epoch
        DistillationMetrics train_metrics = train_epoch();

        // Validate
        DistillationMetrics val_metrics;
        if (val_dataset_ && (epoch + 1) % config_.validation_frequency == 0) {
            val_dataset_->reset();
            val_metrics = validate();
        }

        // Update learning rate
        update_learning_rate();

        // Check early stopping
        float eval_loss = val_dataset_ ? val_metrics.total_loss : train_metrics.total_loss;
        if (config_.early_stopping_patience > 0) {
            if (check_early_stopping(eval_loss)) {
                // Early stopping triggered
                if (callback) {
                    callback(epoch, train_metrics, val_metrics, state_);
                }
                break;
            }
        } else if (eval_loss < state_.best_loss) {
            state_.best_loss = eval_loss;
            save_best_model();
        }

        // Callback
        if (callback) {
            callback(epoch, train_metrics, val_metrics, state_);
        }
    }

    return best_model_;
}

DistillationMetrics DistillationTrainer::train_epoch() {
    DistillationMetrics metrics;
    size_t dataset_size = train_dataset_->size();
    size_t num_batches = (dataset_size + config_.batch_size - 1) / config_.batch_size;

    for (size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
        size_t start = batch_idx * config_.batch_size;
        size_t actual_batch_size = std::min(config_.batch_size, dataset_size - start);

        auto batch = train_dataset_->get_batch(start, actual_batch_size);
        auto batch_metrics = process_batch(batch, true);

        // Accumulate metrics
        metrics.total_loss += batch_metrics.total_loss * batch.batch_size;
        metrics.soft_loss += batch_metrics.soft_loss * batch.batch_size;
        metrics.hard_loss += batch_metrics.hard_loss * batch.batch_size;
        metrics.feature_loss += batch_metrics.feature_loss * batch.batch_size;
        metrics.attention_loss += batch_metrics.attention_loss * batch.batch_size;
        metrics.num_samples += batch.batch_size;

        state_.total_steps++;
    }

    // Average metrics
    if (metrics.num_samples > 0) {
        metrics.total_loss /= metrics.num_samples;
        metrics.soft_loss /= metrics.num_samples;
        metrics.hard_loss /= metrics.num_samples;
        metrics.feature_loss /= metrics.num_samples;
        metrics.attention_loss /= metrics.num_samples;
    }

    return metrics;
}

DistillationMetrics DistillationTrainer::validate() {
    if (!val_dataset_) {
        return DistillationMetrics();
    }

    DistillationMetrics metrics;
    size_t dataset_size = val_dataset_->size();
    size_t num_batches = (dataset_size + config_.batch_size - 1) / config_.batch_size;

    for (size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
        size_t start = batch_idx * config_.batch_size;
        size_t actual_batch_size = std::min(config_.batch_size, dataset_size - start);

        auto batch = val_dataset_->get_batch(start, actual_batch_size);
        auto batch_metrics = process_batch(batch, false);

        metrics.total_loss += batch_metrics.total_loss * batch.batch_size;
        metrics.soft_loss += batch_metrics.soft_loss * batch.batch_size;
        metrics.hard_loss += batch_metrics.hard_loss * batch.batch_size;
        metrics.num_samples += batch.batch_size;
    }

    if (metrics.num_samples > 0) {
        metrics.total_loss /= metrics.num_samples;
        metrics.soft_loss /= metrics.num_samples;
        metrics.hard_loss /= metrics.num_samples;
    }

    return metrics;
}

DistillationMetrics DistillationTrainer::process_batch(
    const DistillationBatch& batch, bool training)
{
    DistillationMetrics metrics;
    metrics.num_samples = batch.batch_size;

    if (batch.inputs.empty()) {
        return metrics;
    }

    // Get teacher outputs
    auto teacher_outputs = models_.teacher->run({}, batch.inputs);

    // Get student outputs
    auto student_outputs = models_.student->run({}, batch.inputs);

    // Compute distillation loss
    std::string output_name = config_.output_name;
    if (teacher_outputs.count(output_name) && student_outputs.count(output_name)) {
        metrics.soft_loss = compute_distillation_loss(
            teacher_outputs[output_name],
            student_outputs[output_name],
            config_.temperature);

        metrics.total_loss = config_.alpha * metrics.soft_loss;
    }

    // Hard loss (if labels available)
    if (batch.labels.has_value() && student_outputs.count(output_name)) {
        metrics.hard_loss = compute_hard_loss(
            student_outputs[output_name],
            batch.labels.value());

        metrics.total_loss += (1.0f - config_.alpha) * metrics.hard_loss;
    }

    // Feature distillation loss
    if (config_.use_feature_distillation && !config_.teacher_layers.empty()) {
        std::vector<Tensor> teacher_features;
        std::vector<Tensor> student_features;

        for (const auto& layer : config_.teacher_layers) {
            if (teacher_outputs.count(layer)) {
                teacher_features.push_back(teacher_outputs[layer]);
            }
        }

        for (const auto& layer : config_.student_layers) {
            if (student_outputs.count(layer)) {
                student_features.push_back(student_outputs[layer]);
            }
        }

        if (!teacher_features.empty() && !student_features.empty()) {
            metrics.feature_loss = compute_feature_loss(teacher_features, student_features);
            metrics.total_loss += config_.feature_loss_weight * metrics.feature_loss;
        }
    }

    // Note: In a real implementation, we would perform backpropagation and
    // weight updates here during training. This simplified version only
    // computes the forward pass and losses.

    return metrics;
}

void DistillationTrainer::update_learning_rate() {
    for (size_t milestone : config_.lr_decay_epochs) {
        if (state_.current_epoch == milestone) {
            state_.current_lr *= config_.lr_decay;
            break;
        }
    }
}

bool DistillationTrainer::check_early_stopping(float loss) {
    if (loss < state_.best_loss) {
        state_.best_loss = loss;
        state_.steps_without_improvement = 0;
        save_best_model();
        return false;
    }

    state_.steps_without_improvement++;
    return state_.steps_without_improvement >= config_.early_stopping_patience;
}

void DistillationTrainer::save_best_model() {
    best_model_ = models_.student_graph;
}

bool DistillationTrainer::is_complete() const {
    return state_.current_epoch >= config_.epochs;
}

// ============================================================================
// Convenience Functions
// ============================================================================

Graph distill_model(
    const Graph& teacher,
    const Graph& student,
    std::shared_ptr<DistillationDataset> dataset,
    const DistillationConfig& config)
{
    auto teacher_session = std::make_shared<InferenceSession>(teacher);
    auto student_session = std::make_shared<InferenceSession>(student);

    DistillationTrainer trainer(config);
    trainer.set_teacher(teacher_session);
    trainer.set_student(student_session, student);
    trainer.set_train_dataset(dataset);

    return trainer.train();
}

Graph distill_model(
    const std::string& teacher_path,
    const std::string& student_path,
    std::shared_ptr<DistillationDataset> dataset,
    const DistillationConfig& config)
{
    DistillationTrainer trainer(config);
    trainer.set_teacher(teacher_path);
    trainer.set_student(student_path);
    trainer.set_train_dataset(dataset);

    return trainer.train();
}

Graph auto_distill(
    const Graph& teacher,
    std::shared_ptr<DistillationDataset> dataset,
    float compression_ratio,
    const DistillationConfig& config)
{
    // Create compressed student architecture
    StudentArchitectureConfig arch_config;
    arch_config.compression_ratio = compression_ratio;

    Graph student = create_student_architecture(teacher, arch_config);

    return distill_model(teacher, student, dataset, config);
}

} // namespace distillation
} // namespace pyflame_rt
