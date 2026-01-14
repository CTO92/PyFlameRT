#include "pyflame_rt/distillation/distillation.hpp"
#include <algorithm>
#include <random>
#include <numeric>

namespace pyflame_rt {
namespace distillation {

// ============================================================================
// InMemoryDataset Implementation
// ============================================================================

InMemoryDataset::InMemoryDataset(
    const std::vector<std::unordered_map<std::string, Tensor>>& inputs,
    const std::vector<Tensor>& labels)
    : inputs_(inputs), labels_(labels)
{
    has_labels_ = !labels_.empty();
    indices_.resize(inputs_.size());
    std::iota(indices_.begin(), indices_.end(), 0);
}

void InMemoryDataset::add_sample(
    const std::unordered_map<std::string, Tensor>& inputs,
    const Tensor& label)
{
    inputs_.push_back(inputs);
    if (label.num_elements() > 0) {
        labels_.push_back(label);
        has_labels_ = true;
    }
    indices_.push_back(indices_.size());
}

void InMemoryDataset::add_samples(
    const Tensor& inputs,
    const Tensor& labels,
    const std::string& input_name)
{
    // Assume inputs is [batch, ...] and labels is [batch] or [batch, ...]
    if (inputs.shape().empty()) return;

    size_t batch_size = inputs.shape()[0];
    // Security: check for zero batch size (HIGH-D4 fix)
    if (batch_size == 0) return;

    size_t sample_size = inputs.num_elements() / batch_size;
    // Security: check for zero sample size (HIGH-D4 fix)
    if (sample_size == 0) return;

    const float* input_data = static_cast<const float*>(inputs.data());
    // Security: null pointer check (HIGH-D4 fix)
    if (!input_data) {
        throw std::runtime_error("Null data pointer in inputs tensor");
    }

    const float* label_data = labels.num_elements() > 0 ?
        static_cast<const float*>(labels.data()) : nullptr;

    // Compute sample shape
    std::vector<int64_t> sample_shape(inputs.shape().begin() + 1, inputs.shape().end());
    if (sample_shape.empty()) {
        sample_shape = {1};
    }

    size_t label_size = 0;
    std::vector<int64_t> label_shape;
    if (label_data && labels.shape().size() > 0) {
        label_size = labels.num_elements() / batch_size;
        label_shape = std::vector<int64_t>(labels.shape().begin() + 1, labels.shape().end());
        if (label_shape.empty()) {
            label_shape = {1};
        }
        has_labels_ = true;
    }

    for (size_t i = 0; i < batch_size; ++i) {
        // Create input tensor for this sample
        Tensor sample_input(sample_shape, inputs.dtype());
        float* dst = static_cast<float*>(sample_input.data());
        // Security: null pointer check (HIGH-D5 fix)
        if (!dst) {
            throw std::runtime_error("Null data pointer in sample tensor");
        }
        std::copy(input_data + i * sample_size,
                  input_data + (i + 1) * sample_size,
                  dst);

        std::unordered_map<std::string, Tensor> sample_inputs;
        sample_inputs[input_name] = std::move(sample_input);
        inputs_.push_back(std::move(sample_inputs));

        // Create label tensor for this sample
        if (label_data) {
            Tensor sample_label(label_shape, labels.dtype());
            float* label_dst = static_cast<float*>(sample_label.data());
            // Security: null pointer check (HIGH-D5 fix)
            if (!label_dst) {
                throw std::runtime_error("Null data pointer in label tensor");
            }
            std::copy(label_data + i * label_size,
                      label_data + (i + 1) * label_size,
                      label_dst);
            labels_.push_back(std::move(sample_label));
        }

        indices_.push_back(indices_.size());
    }
}

size_t InMemoryDataset::size() const {
    return inputs_.size();
}

DistillationBatch InMemoryDataset::get_batch(size_t start, size_t batch_size) {
    DistillationBatch batch;

    if (start >= inputs_.size()) {
        batch.batch_size = 0;
        return batch;
    }

    size_t actual_size = std::min(batch_size, inputs_.size() - start);
    batch.batch_size = actual_size;

    if (actual_size == 0) {
        return batch;
    }

    // For simplicity, we'll just return the first sample's inputs
    // A real implementation would batch multiple samples together
    size_t idx = indices_[start];
    batch.inputs = inputs_[idx];

    if (has_labels_ && idx < labels_.size()) {
        batch.labels = labels_[idx];
    }

    // If batch size > 1, we should concatenate tensors
    // This is a simplified implementation that processes one sample at a time
    if (actual_size > 1) {
        // Get the first input name and shape
        if (!batch.inputs.empty()) {
            auto& first_input = batch.inputs.begin()->second;
            auto input_shape = first_input.shape();

            // Create batched input shape
            std::vector<int64_t> batched_shape;
            batched_shape.push_back(static_cast<int64_t>(actual_size));
            batched_shape.insert(batched_shape.end(), input_shape.begin(), input_shape.end());

            // Create batched tensors
            std::unordered_map<std::string, Tensor> batched_inputs;
            for (const auto& [name, tensor] : inputs_[indices_[start]]) {
                auto shape = tensor.shape();
                std::vector<int64_t> batch_shape;
                batch_shape.push_back(static_cast<int64_t>(actual_size));
                batch_shape.insert(batch_shape.end(), shape.begin(), shape.end());

                Tensor batched(batch_shape, tensor.dtype());
                float* dst = static_cast<float*>(batched.data());

                // Security: null pointer check (HIGH-D6 fix)
                if (!dst) continue;

                size_t sample_size = tensor.num_elements();
                // Security: skip if sample_size is zero (HIGH-D6 fix)
                if (sample_size == 0) continue;

                for (size_t i = 0; i < actual_size; ++i) {
                    size_t sample_idx = indices_[start + i];
                    if (sample_idx < inputs_.size() && inputs_[sample_idx].count(name)) {
                        const float* src = static_cast<const float*>(
                            inputs_[sample_idx].at(name).data());
                        // Security: null pointer check (HIGH-D6 fix)
                        if (src) {
                            std::copy(src, src + sample_size, dst + i * sample_size);
                        }
                    }
                }

                batched_inputs[name] = std::move(batched);
            }
            batch.inputs = std::move(batched_inputs);
        }

        // Batch labels
        if (has_labels_ && !labels_.empty()) {
            size_t first_idx = indices_[start];
            if (first_idx < labels_.size()) {
                auto label_shape = labels_[first_idx].shape();
                std::vector<int64_t> batch_shape;
                batch_shape.push_back(static_cast<int64_t>(actual_size));
                batch_shape.insert(batch_shape.end(), label_shape.begin(), label_shape.end());

                Tensor batched_labels(batch_shape, labels_[first_idx].dtype());
                float* dst = static_cast<float*>(batched_labels.data());

                // Security: null pointer check (HIGH-D6 fix)
                if (dst) {
                    size_t label_size = labels_[first_idx].num_elements();
                    // Security: skip if label_size is zero (HIGH-D6 fix)
                    if (label_size > 0) {
                        for (size_t i = 0; i < actual_size; ++i) {
                            size_t sample_idx = indices_[start + i];
                            if (sample_idx < labels_.size()) {
                                const float* src = static_cast<const float*>(labels_[sample_idx].data());
                                // Security: null pointer check (HIGH-D6 fix)
                                if (src) {
                                    std::copy(src, src + label_size, dst + i * label_size);
                                }
                            }
                        }
                    }
                    batch.labels = std::move(batched_labels);
                }
            }
        }
    }

    return batch;
}

void InMemoryDataset::shuffle() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::shuffle(indices_.begin(), indices_.end(), gen);
}

void InMemoryDataset::reset() {
    // Nothing to do for in-memory dataset
}

} // namespace distillation
} // namespace pyflame_rt
