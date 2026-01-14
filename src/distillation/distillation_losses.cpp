#include "pyflame_rt/distillation/distillation.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace pyflame_rt {
namespace distillation {

Tensor DistillationTrainer::softmax_with_temperature(
    const Tensor& logits, float temperature)
{
    if (logits.shape().size() < 1) {
        throw std::invalid_argument("Logits must have at least 1 dimension");
    }

    // Security: validate temperature is positive (CRIT-D1 fix)
    if (temperature <= 0.0f) {
        throw std::invalid_argument("Temperature must be positive");
    }

    size_t total = logits.num_elements();
    // Security: handle empty tensors (HIGH-D1 fix)
    if (total == 0) {
        return Tensor(logits.shape(), logits.dtype());
    }

    Tensor result(logits.shape(), logits.dtype());
    const float* src = static_cast<const float*>(logits.data());
    float* dst = static_cast<float*>(result.data());

    // Security: null pointer checks (HIGH-D1 fix)
    if (!src || !dst) {
        throw std::runtime_error("Null data pointer in tensor");
    }

    // Assume last dimension is the class dimension
    size_t num_classes = logits.shape().back();
    // Security: check for zero dimensions (CRIT-D1 fix)
    if (num_classes == 0) {
        throw std::invalid_argument("Number of classes cannot be zero");
    }
    size_t num_samples = total / num_classes;

    for (size_t s = 0; s < num_samples; ++s) {
        const float* sample_src = src + s * num_classes;
        float* sample_dst = dst + s * num_classes;

        // Find max for numerical stability
        float max_val = sample_src[0];
        for (size_t c = 1; c < num_classes; ++c) {
            max_val = std::max(max_val, sample_src[c]);
        }

        // Compute softmax with temperature
        float sum = 0.0f;
        for (size_t c = 0; c < num_classes; ++c) {
            float val = std::exp((sample_src[c] - max_val) / temperature);
            sample_dst[c] = val;
            sum += val;
        }

        // Normalize
        if (sum > 0.0f) {
            for (size_t c = 0; c < num_classes; ++c) {
                sample_dst[c] /= sum;
            }
        }
    }

    return result;
}

Tensor DistillationTrainer::kl_divergence(const Tensor& p, const Tensor& q) {
    // KL(p || q) = sum(p * log(p / q))
    // Input tensors should be probability distributions (already softmaxed)

    if (p.num_elements() != q.num_elements()) {
        throw std::invalid_argument("KL divergence requires same-sized tensors");
    }

    size_t num_elements = p.num_elements();
    // Security: handle empty tensors (HIGH-D2 fix)
    if (num_elements == 0) {
        Tensor result({1}, DType::Float32);
        float* out = static_cast<float*>(result.data());
        *out = 0.0f;
        return result;
    }

    Tensor result({1}, DType::Float32);
    float* out = static_cast<float*>(result.data());

    const float* p_data = static_cast<const float*>(p.data());
    const float* q_data = static_cast<const float*>(q.data());

    // Security: null pointer checks (HIGH-D2 fix)
    if (!p_data || !q_data || !out) {
        throw std::runtime_error("Null data pointer in tensor");
    }

    float kl = 0.0f;
    const float epsilon = 1e-10f;

    for (size_t i = 0; i < num_elements; ++i) {
        float p_val = std::max(p_data[i], epsilon);
        float q_val = std::max(q_data[i], epsilon);
        kl += p_val * std::log(p_val / q_val);
    }

    // Average over batch if applicable
    if (p.shape().empty()) {
        *out = kl;
        return result;
    }
    size_t num_classes = p.shape().back();
    // Security: check for zero dimensions (HIGH-D2 fix)
    if (num_classes == 0) {
        *out = 0.0f;
        return result;
    }
    size_t batch_size = num_elements / num_classes;
    // Security: prevent division by zero (HIGH-D2 fix)
    if (batch_size == 0) {
        batch_size = 1;
    }
    *out = kl / batch_size;

    return result;
}

float DistillationTrainer::compute_distillation_loss(
    const Tensor& teacher_logits,
    const Tensor& student_logits,
    float temperature)
{
    // Soft targets from teacher
    Tensor teacher_soft = softmax_with_temperature(teacher_logits, temperature);

    // Student predictions with temperature
    Tensor student_soft = softmax_with_temperature(student_logits, temperature);

    // KL divergence loss
    Tensor kl = kl_divergence(teacher_soft, student_soft);

    // Scale by T^2 as per Hinton et al. "Distilling the Knowledge in a Neural Network"
    float kl_val = static_cast<const float*>(kl.data())[0];
    return kl_val * temperature * temperature;
}

float DistillationTrainer::compute_feature_loss(
    const std::vector<Tensor>& teacher_features,
    const std::vector<Tensor>& student_features)
{
    if (teacher_features.empty() || student_features.empty()) {
        return 0.0f;
    }

    float total_loss = 0.0f;
    size_t num_pairs = std::min(teacher_features.size(), student_features.size());

    for (size_t i = 0; i < num_pairs; ++i) {
        const Tensor& t_feat = teacher_features[i];
        const Tensor& s_feat = student_features[i];

        // Use MSE loss for feature matching
        total_loss += mse_loss(t_feat, s_feat);
    }

    return total_loss / num_pairs;
}

float DistillationTrainer::compute_attention_loss(
    const std::vector<Tensor>& teacher_attention,
    const std::vector<Tensor>& student_attention)
{
    if (teacher_attention.empty() || student_attention.empty()) {
        return 0.0f;
    }

    float total_loss = 0.0f;
    size_t num_pairs = std::min(teacher_attention.size(), student_attention.size());

    for (size_t i = 0; i < num_pairs; ++i) {
        const Tensor& t_attn = teacher_attention[i];
        const Tensor& s_attn = student_attention[i];

        // Attention transfer: L2 norm of difference
        total_loss += mse_loss(t_attn, s_attn);
    }

    return total_loss / num_pairs;
}

float DistillationTrainer::compute_hard_loss(
    const Tensor& logits,
    const Tensor& labels)
{
    // Cross-entropy loss
    // logits: [batch_size, num_classes]
    // labels: [batch_size] (class indices) or [batch_size, num_classes] (one-hot)

    // Security: handle empty tensors (HIGH-D3 fix)
    if (logits.num_elements() == 0 || labels.num_elements() == 0) {
        return 0.0f;
    }

    // Security: validate shape (HIGH-D3 fix)
    if (logits.shape().empty()) {
        throw std::invalid_argument("Logits must have at least 1 dimension");
    }

    size_t num_classes = logits.shape().back();
    // Security: check for zero dimensions (CRIT-D2 fix)
    if (num_classes == 0) {
        throw std::invalid_argument("Number of classes cannot be zero");
    }
    size_t batch_size = logits.num_elements() / num_classes;
    if (batch_size == 0) {
        return 0.0f;
    }

    const float* label_data = static_cast<const float*>(labels.data());
    // Security: null pointer check (HIGH-D3 fix)
    if (!label_data) {
        throw std::runtime_error("Null data pointer in labels tensor");
    }

    // Apply softmax to logits
    Tensor probs = softmax_with_temperature(logits, 1.0f);
    const float* prob_data = static_cast<const float*>(probs.data());
    // Security: null pointer check (HIGH-D3 fix)
    if (!prob_data) {
        throw std::runtime_error("Null data pointer in probs tensor");
    }

    float loss = 0.0f;
    const float epsilon = 1e-10f;

    if (labels.num_elements() == batch_size) {
        // Labels are class indices
        for (size_t b = 0; b < batch_size; ++b) {
            // Security: validate label index (HIGH-D3 fix)
            float label_val = label_data[b];
            if (label_val < 0.0f || label_val >= static_cast<float>(num_classes)) {
                continue;  // Skip invalid labels
            }
            size_t label_idx = static_cast<size_t>(label_val);
            float p = std::max(prob_data[b * num_classes + label_idx], epsilon);
            loss -= std::log(p);
        }
    } else {
        // Labels are one-hot or soft targets
        size_t label_elements = labels.num_elements();
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t c = 0; c < num_classes; ++c) {
                size_t idx = b * num_classes + c;
                // Security: bounds check (HIGH-D3 fix)
                if (idx >= label_elements) break;
                float target = label_data[idx];
                if (target > 0.0f) {
                    float p = std::max(prob_data[idx], epsilon);
                    loss -= target * std::log(p);
                }
            }
        }
    }

    return loss / batch_size;
}

float DistillationTrainer::mse_loss(const Tensor& a, const Tensor& b) {
    size_t n = std::min(a.num_elements(), b.num_elements());
    if (n == 0) return 0.0f;

    const float* a_data = static_cast<const float*>(a.data());
    const float* b_data = static_cast<const float*>(b.data());

    // Security: null pointer checks (MED-D1 fix)
    if (!a_data || !b_data) {
        throw std::runtime_error("Null data pointer in tensor");
    }

    float mse = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float diff = a_data[i] - b_data[i];
        mse += diff * diff;
    }

    return mse / n;
}

float DistillationTrainer::cosine_loss(const Tensor& a, const Tensor& b) {
    size_t n = std::min(a.num_elements(), b.num_elements());
    if (n == 0) return 0.0f;

    const float* a_data = static_cast<const float*>(a.data());
    const float* b_data = static_cast<const float*>(b.data());

    // Security: null pointer checks (MED-D2 fix)
    if (!a_data || !b_data) {
        throw std::runtime_error("Null data pointer in tensor");
    }

    float dot = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;

    for (size_t i = 0; i < n; ++i) {
        dot += a_data[i] * b_data[i];
        norm_a += a_data[i] * a_data[i];
        norm_b += b_data[i] * b_data[i];
    }

    norm_a = std::sqrt(norm_a);
    norm_b = std::sqrt(norm_b);

    const float epsilon = 1e-10f;
    float cosine_sim = dot / (norm_a * norm_b + epsilon);

    // Return 1 - cosine_similarity as loss (0 = identical, 2 = opposite)
    return 1.0f - cosine_sim;
}

} // namespace distillation
} // namespace pyflame_rt
