#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include "pyflame_rt/distillation/distillation.hpp"

namespace py = pybind11;
using namespace pyflame_rt;
using namespace pyflame_rt::distillation;

void bind_distillation(py::module_& m) {
    auto distill = m.def_submodule("distillation",
        "Knowledge distillation for model compression");

    // ========================================================================
    // Enums
    // ========================================================================

    py::enum_<DistillationLoss>(distill, "DistillationLoss",
        "Loss function for knowledge distillation")
        .value("KL_DIVERGENCE", DistillationLoss::KLDivergence,
            "KL divergence between soft targets")
        .value("MSE", DistillationLoss::MSE,
            "Mean squared error for feature matching")
        .value("COSINE", DistillationLoss::Cosine,
            "Cosine similarity loss")
        .value("ATTENTION", DistillationLoss::Attention,
            "Attention transfer loss")
        .value("HINT", DistillationLoss::Hint,
            "Hint-based feature distillation");

    py::enum_<StudentInitStrategy>(distill, "StudentInitStrategy",
        "Strategy for initializing student model")
        .value("RANDOM", StudentInitStrategy::Random,
            "Random initialization")
        .value("TEACHER_SUBSET", StudentInitStrategy::TeacherSubset,
            "Initialize from subset of teacher weights")
        .value("PRETRAINED", StudentInitStrategy::Pretrained,
            "Use pretrained weights");

    // ========================================================================
    // DistillationConfig
    // ========================================================================

    py::class_<DistillationConfig>(distill, "DistillationConfig",
        "Configuration for knowledge distillation")
        .def(py::init<>())
        .def_readwrite("temperature", &DistillationConfig::temperature,
            "Temperature for softmax (default 4.0)")
        .def_readwrite("alpha", &DistillationConfig::alpha,
            "Weight for distillation loss (vs hard label loss)")
        .def_readwrite("loss_type", &DistillationConfig::loss_type,
            "Type of distillation loss")
        .def_readwrite("use_hard_labels", &DistillationConfig::use_hard_labels,
            "Whether to also use hard labels")
        .def_readwrite("hard_label_weight", &DistillationConfig::hard_label_weight,
            "Weight for hard label loss")
        .def_readwrite("feature_layers", &DistillationConfig::feature_layers,
            "Layer names for feature distillation")
        .def_readwrite("feature_weights", &DistillationConfig::feature_weights,
            "Weights for feature distillation losses")
        .def_readwrite("attention_layers", &DistillationConfig::attention_layers,
            "Layer names for attention transfer")
        .def_readwrite("normalize_features", &DistillationConfig::normalize_features,
            "Whether to normalize features before loss")
        .def_readwrite("student_init", &DistillationConfig::student_init,
            "Student initialization strategy")
        .def_static("soft_label", &DistillationConfig::soft_label,
            py::arg("temperature") = 4.0f,
            "Create config for soft label distillation")
        .def_static("feature_distillation", &DistillationConfig::feature_distillation,
            py::arg("layers"),
            "Create config for feature distillation")
        .def_static("attention_transfer", &DistillationConfig::attention_transfer,
            py::arg("layers"),
            "Create config for attention transfer");

    // ========================================================================
    // StudentConfig
    // ========================================================================

    py::class_<StudentConfig>(distill, "StudentConfig",
        "Configuration for student model architecture")
        .def(py::init<>())
        .def_readwrite("hidden_dim_ratio", &StudentConfig::hidden_dim_ratio,
            "Ratio of student hidden dim to teacher")
        .def_readwrite("num_layers_ratio", &StudentConfig::num_layers_ratio,
            "Ratio of student layers to teacher")
        .def_readwrite("num_heads_ratio", &StudentConfig::num_heads_ratio,
            "Ratio of attention heads (for transformers)")
        .def_readwrite("use_layer_subset", &StudentConfig::use_layer_subset,
            "Use subset of teacher layers instead of new layers")
        .def_readwrite("layer_indices", &StudentConfig::layer_indices,
            "Indices of teacher layers to keep")
        .def_readwrite("custom_architecture", &StudentConfig::custom_architecture,
            "Use custom student architecture")
        .def_static("half_size", &StudentConfig::half_size,
            "Create config for half-size student")
        .def_static("quarter_size", &StudentConfig::quarter_size,
            "Create config for quarter-size student");

    // ========================================================================
    // TrainingConfig
    // ========================================================================

    py::class_<TrainingConfig>(distill, "TrainingConfig",
        "Configuration for distillation training")
        .def(py::init<>())
        .def_readwrite("learning_rate", &TrainingConfig::learning_rate,
            "Initial learning rate")
        .def_readwrite("batch_size", &TrainingConfig::batch_size,
            "Training batch size")
        .def_readwrite("num_epochs", &TrainingConfig::num_epochs,
            "Number of training epochs")
        .def_readwrite("warmup_steps", &TrainingConfig::warmup_steps,
            "Number of warmup steps")
        .def_readwrite("weight_decay", &TrainingConfig::weight_decay,
            "Weight decay for regularization")
        .def_readwrite("lr_schedule", &TrainingConfig::lr_schedule,
            "Learning rate schedule (cosine, linear, constant)")
        .def_readwrite("gradient_clip", &TrainingConfig::gradient_clip,
            "Gradient clipping norm")
        .def_readwrite("early_stopping_patience", &TrainingConfig::early_stopping_patience,
            "Patience for early stopping")
        .def_readwrite("checkpoint_frequency", &TrainingConfig::checkpoint_frequency,
            "Checkpoint every N steps");

    // ========================================================================
    // DistillationDataset
    // ========================================================================

    py::class_<DistillationDataset, std::shared_ptr<DistillationDataset>>(
        distill, "DistillationDataset",
        "Abstract dataset for distillation training")
        .def("size", &DistillationDataset::size,
            "Get dataset size")
        .def("get_batch", &DistillationDataset::get_batch,
            py::arg("start"), py::arg("batch_size"),
            "Get a batch of samples");

    py::class_<InMemoryDataset, DistillationDataset,
        std::shared_ptr<InMemoryDataset>>(
        distill, "InMemoryDataset",
        "In-memory dataset for distillation")
        .def(py::init<>())
        .def("add_sample", &InMemoryDataset::add_sample,
            py::arg("inputs"), py::arg("labels") = std::unordered_map<std::string, Tensor>{},
            "Add a training sample")
        .def("clear", &InMemoryDataset::clear,
            "Clear all samples")
        .def("shuffle", &InMemoryDataset::shuffle,
            "Shuffle the dataset");

    // ========================================================================
    // DistillationResult
    // ========================================================================

    py::class_<DistillationResult>(distill, "DistillationResult",
        "Result of distillation training")
        .def_readonly("student_graph", &DistillationResult::student_graph,
            "Trained student model graph")
        .def_readonly("final_loss", &DistillationResult::final_loss,
            "Final training loss")
        .def_readonly("best_loss", &DistillationResult::best_loss,
            "Best validation loss achieved")
        .def_readonly("training_steps", &DistillationResult::training_steps,
            "Total training steps")
        .def_readonly("loss_history", &DistillationResult::loss_history,
            "Training loss history")
        .def_readonly("teacher_size_bytes", &DistillationResult::teacher_size_bytes,
            "Teacher model size")
        .def_readonly("student_size_bytes", &DistillationResult::student_size_bytes,
            "Student model size")
        .def("compression_ratio", &DistillationResult::compression_ratio,
            "Get compression ratio");

    // ========================================================================
    // DistillationTrainer
    // ========================================================================

    py::class_<DistillationTrainer>(distill, "DistillationTrainer",
        "Trainer for knowledge distillation")
        .def(py::init<const Graph&, const Graph&, const DistillationConfig&>(),
            py::arg("teacher"), py::arg("student"), py::arg("config"),
            "Create trainer with teacher and student models")
        .def(py::init<const Graph&, const StudentConfig&, const DistillationConfig&>(),
            py::arg("teacher"), py::arg("student_config"), py::arg("config"),
            "Create trainer that generates student from config")
        .def("set_training_config", &DistillationTrainer::set_training_config,
            py::arg("config"),
            "Set training configuration")
        .def("set_dataset", &DistillationTrainer::set_dataset,
            py::arg("dataset"),
            "Set training dataset")
        .def("set_validation_dataset", &DistillationTrainer::set_validation_dataset,
            py::arg("dataset"),
            "Set validation dataset")
        .def("train", &DistillationTrainer::train,
            py::arg("callback") = nullptr,
            "Run distillation training")
        .def("train_step", &DistillationTrainer::train_step,
            py::arg("inputs"), py::arg("labels"),
            "Run a single training step")
        .def("evaluate", &DistillationTrainer::evaluate,
            py::arg("inputs"),
            "Evaluate student on inputs")
        .def("get_student", &DistillationTrainer::get_student,
            "Get current student model")
        .def("save_checkpoint", &DistillationTrainer::save_checkpoint,
            py::arg("path"),
            "Save training checkpoint")
        .def("load_checkpoint", &DistillationTrainer::load_checkpoint,
            py::arg("path"),
            "Load training checkpoint");

    // ========================================================================
    // Utility Functions
    // ========================================================================

    distill.def("create_student_model", &create_student_model,
        py::arg("teacher"), py::arg("config"),
        "Create student model from teacher");

    distill.def("compute_soft_targets", &compute_soft_targets,
        py::arg("logits"), py::arg("temperature"),
        "Compute soft targets with temperature scaling");

    distill.def("kl_divergence_loss", &kl_divergence_loss,
        py::arg("student_logits"), py::arg("teacher_logits"), py::arg("temperature"),
        "Compute KL divergence loss for soft targets");

    distill.def("feature_loss", &feature_loss,
        py::arg("student_features"), py::arg("teacher_features"),
        py::arg("loss_type") = DistillationLoss::MSE,
        "Compute feature distillation loss");

    distill.def("distill", &distill_model,
        py::arg("teacher"), py::arg("dataset"),
        py::arg("compression_ratio") = 0.5f,
        py::arg("num_epochs") = 10,
        "Convenience function for basic distillation");
}
