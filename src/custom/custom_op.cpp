#include "pyflame_rt/custom/custom_op.hpp"
#include <stdexcept>
#include <sstream>

namespace pyflame_rt {
namespace custom {

// ============================================================================
// CustomOp Implementation
// ============================================================================

CustomOp::CustomOp(const OpSchema& schema)
    : schema_(schema)
{
}

CustomOp::~CustomOp() = default;

CustomOp& CustomOp::register_kernel(BackendType backend, KernelFunc kernel) {
    if (finalized_) {
        throw std::runtime_error("Cannot register kernel after finalization");
    }
    if (!kernel) {
        throw std::invalid_argument("Kernel function cannot be null");
    }
    kernels_[backend] = std::move(kernel);
    return *this;
}

CustomOp& CustomOp::register_kernel_multi(BackendType backend,
                                           MultiOutputKernelFunc kernel) {
    if (finalized_) {
        throw std::runtime_error("Cannot register kernel after finalization");
    }
    if (!kernel) {
        throw std::invalid_argument("Kernel function cannot be null");
    }
    multi_kernels_[backend] = std::move(kernel);
    return *this;
}

CustomOp& CustomOp::register_shape_inference(ShapeInferenceFunc func) {
    if (finalized_) {
        throw std::runtime_error("Cannot register shape inference after finalization");
    }
    shape_inference_ = std::move(func);
    return *this;
}

CustomOp& CustomOp::register_type_inference(TypeInferenceFunc func) {
    if (finalized_) {
        throw std::runtime_error("Cannot register type inference after finalization");
    }
    type_inference_ = std::move(func);
    return *this;
}

CustomOp& CustomOp::register_gradient(GradientFunc func) {
    if (finalized_) {
        throw std::runtime_error("Cannot register gradient after finalization");
    }
    gradient_ = std::move(func);
    return *this;
}

void CustomOp::finalize() {
    // Validate schema
    if (schema_.name.empty()) {
        throw std::runtime_error("Custom op must have a name");
    }

    if (kernels_.empty() && multi_kernels_.empty()) {
        throw std::runtime_error("Custom op must have at least one kernel");
    }

    // Register with global operator registry
    auto full = full_name();

    // Create a wrapper that captures this custom op
    OperatorRegistry::instance().register_op(full,
        [this](const std::vector<Tensor>& inputs,
               const std::unordered_map<std::string, std::any>& attrs) {
            return execute(inputs, attrs);
        });

    finalized_ = true;
}

Tensor CustomOp::execute(
    const std::vector<Tensor>& inputs,
    const std::unordered_map<std::string, std::any>& attrs,
    BackendType backend)
{
    // Security: validate input count against schema (CRIT-C1 fix)
    size_t required_inputs = 0;
    for (const auto& input_spec : schema_.inputs) {
        if (!input_spec.optional) {
            required_inputs++;
        }
    }
    if (inputs.size() < required_inputs) {
        throw std::invalid_argument(
            "Custom op '" + schema_.name + "' requires at least " +
            std::to_string(required_inputs) + " inputs, got " +
            std::to_string(inputs.size()));
    }

    // Security: validate required attributes (HIGH-C1 fix)
    for (const auto& attr_spec : schema_.attributes) {
        if (attr_spec.required && attrs.find(attr_spec.name) == attrs.end()) {
            throw std::invalid_argument(
                "Custom op '" + schema_.name + "' requires attribute: " + attr_spec.name);
        }
    }

    // Find kernel for backend
    auto it = kernels_.find(backend);
    if (it == kernels_.end()) {
        it = kernels_.find(BackendType::All);
    }
    if (it == kernels_.end() && backend != BackendType::CPU) {
        // Fallback to CPU
        it = kernels_.find(BackendType::CPU);
    }

    if (it != kernels_.end()) {
        return it->second(inputs, attrs);
    }

    // Try multi-output kernel
    auto multi_it = multi_kernels_.find(backend);
    if (multi_it == multi_kernels_.end()) {
        multi_it = multi_kernels_.find(BackendType::All);
    }
    if (multi_it == multi_kernels_.end() && backend != BackendType::CPU) {
        multi_it = multi_kernels_.find(BackendType::CPU);
    }

    if (multi_it != multi_kernels_.end()) {
        auto results = multi_it->second(inputs, attrs);
        if (results.empty()) {
            throw std::runtime_error("Multi-output kernel returned no outputs");
        }
        return results[0];
    }

    throw std::runtime_error("No kernel registered for backend in custom op: " + schema_.name);
}

std::vector<Tensor> CustomOp::execute_multi(
    const std::vector<Tensor>& inputs,
    const std::unordered_map<std::string, std::any>& attrs,
    BackendType backend)
{
    // Security: validate input count against schema (HIGH-C2 fix)
    size_t required_inputs = 0;
    for (const auto& input_spec : schema_.inputs) {
        if (!input_spec.optional) {
            required_inputs++;
        }
    }
    if (inputs.size() < required_inputs) {
        throw std::invalid_argument(
            "Custom op '" + schema_.name + "' requires at least " +
            std::to_string(required_inputs) + " inputs, got " +
            std::to_string(inputs.size()));
    }

    // Security: validate required attributes (HIGH-C2 fix)
    for (const auto& attr_spec : schema_.attributes) {
        if (attr_spec.required && attrs.find(attr_spec.name) == attrs.end()) {
            throw std::invalid_argument(
                "Custom op '" + schema_.name + "' requires attribute: " + attr_spec.name);
        }
    }

    // Try multi-output kernel first
    auto multi_it = multi_kernels_.find(backend);
    if (multi_it == multi_kernels_.end()) {
        multi_it = multi_kernels_.find(BackendType::All);
    }
    if (multi_it == multi_kernels_.end() && backend != BackendType::CPU) {
        multi_it = multi_kernels_.find(BackendType::CPU);
    }

    if (multi_it != multi_kernels_.end()) {
        return multi_it->second(inputs, attrs);
    }

    // Fall back to single output kernel
    return {execute(inputs, attrs, backend)};
}

std::vector<TensorInfo> CustomOp::infer_shapes(
    const std::vector<TensorInfo>& inputs,
    const std::unordered_map<std::string, std::any>& attrs)
{
    if (shape_inference_) {
        return shape_inference_(inputs, attrs);
    }

    // Default: same shape as first input
    std::vector<TensorInfo> outputs;
    for (const auto& output_spec : schema_.outputs) {
        TensorInfo info;
        info.name = output_spec.name;
        info.dtype = output_spec.dtype;
        if (!inputs.empty()) {
            info.shape = inputs[0].shape;
        }
        outputs.push_back(info);
    }
    return outputs;
}

std::vector<DType> CustomOp::infer_types(
    const std::vector<DType>& input_types,
    const std::unordered_map<std::string, std::any>& attrs)
{
    if (type_inference_) {
        return type_inference_(input_types, attrs);
    }

    // Default: same type as first input or schema default
    std::vector<DType> output_types;
    for (const auto& output_spec : schema_.outputs) {
        if (!input_types.empty()) {
            output_types.push_back(input_types[0]);
        } else {
            output_types.push_back(output_spec.dtype);
        }
    }
    return output_types;
}

std::vector<Tensor> CustomOp::compute_gradient(
    const std::vector<Tensor>& grad_outputs,
    const std::vector<Tensor>& inputs,
    const std::vector<Tensor>& outputs,
    const std::unordered_map<std::string, std::any>& attrs)
{
    if (!gradient_) {
        throw std::runtime_error("No gradient function registered for " + schema_.name);
    }
    return gradient_(grad_outputs, inputs, outputs, attrs);
}

bool CustomOp::has_gradient() const {
    return gradient_ != nullptr;
}

std::string CustomOp::full_name() const {
    if (schema_.domain.empty() || schema_.domain == "custom") {
        return schema_.name;
    }
    return schema_.domain + "::" + schema_.name;
}

// ============================================================================
// CustomOpBuilder Implementation
// ============================================================================

CustomOpBuilder::CustomOpBuilder(const std::string& name) {
    schema_.name = name;
    schema_.domain = "custom";
    schema_.version = 1;
}

CustomOpBuilder& CustomOpBuilder::domain(const std::string& domain) {
    schema_.domain = domain;
    return *this;
}

CustomOpBuilder& CustomOpBuilder::version(int version) {
    schema_.version = version;
    return *this;
}

CustomOpBuilder& CustomOpBuilder::input(
    const std::string& name,
    const std::string& description,
    DType dtype,
    bool optional)
{
    OpSchema::InputSpec spec;
    spec.name = name;
    spec.description = description;
    spec.dtype = dtype;
    spec.optional = optional;
    schema_.inputs.push_back(spec);
    return *this;
}

CustomOpBuilder& CustomOpBuilder::output(
    const std::string& name,
    const std::string& description,
    DType dtype)
{
    OpSchema::OutputSpec spec;
    spec.name = name;
    spec.description = description;
    spec.dtype = dtype;
    schema_.outputs.push_back(spec);
    return *this;
}

CustomOpBuilder& CustomOpBuilder::attr_int(
    const std::string& name,
    int default_value,
    const std::string& description,
    bool required)
{
    OpSchema::AttributeSpec spec;
    spec.name = name;
    spec.type = "int";
    spec.default_value = default_value;
    spec.description = description;
    spec.required = required;
    schema_.attributes.push_back(spec);
    return *this;
}

CustomOpBuilder& CustomOpBuilder::attr_float(
    const std::string& name,
    float default_value,
    const std::string& description,
    bool required)
{
    OpSchema::AttributeSpec spec;
    spec.name = name;
    spec.type = "float";
    spec.default_value = default_value;
    spec.description = description;
    spec.required = required;
    schema_.attributes.push_back(spec);
    return *this;
}

CustomOpBuilder& CustomOpBuilder::attr_string(
    const std::string& name,
    const std::string& default_value,
    const std::string& description,
    bool required)
{
    OpSchema::AttributeSpec spec;
    spec.name = name;
    spec.type = "string";
    spec.default_value = default_value;
    spec.description = description;
    spec.required = required;
    schema_.attributes.push_back(spec);
    return *this;
}

CustomOpBuilder& CustomOpBuilder::attr_required(
    const std::string& name,
    const std::string& type,
    const std::string& description)
{
    OpSchema::AttributeSpec spec;
    spec.name = name;
    spec.type = type;
    spec.description = description;
    spec.required = true;
    schema_.attributes.push_back(spec);
    return *this;
}

CustomOpBuilder& CustomOpBuilder::doc(const std::string& doc) {
    schema_.doc = doc;
    return *this;
}

CustomOpBuilder& CustomOpBuilder::kernel(KernelFunc func) {
    kernels_[BackendType::CPU] = std::move(func);
    return *this;
}

CustomOpBuilder& CustomOpBuilder::kernel(BackendType backend, KernelFunc func) {
    kernels_[backend] = std::move(func);
    return *this;
}

CustomOpBuilder& CustomOpBuilder::kernel_multi(MultiOutputKernelFunc func) {
    multi_kernels_[BackendType::CPU] = std::move(func);
    return *this;
}

CustomOpBuilder& CustomOpBuilder::kernel_multi(BackendType backend, MultiOutputKernelFunc func) {
    multi_kernels_[backend] = std::move(func);
    return *this;
}

CustomOpBuilder& CustomOpBuilder::shape_inference(ShapeInferenceFunc func) {
    shape_inference_ = std::move(func);
    return *this;
}

CustomOpBuilder& CustomOpBuilder::type_inference(TypeInferenceFunc func) {
    type_inference_ = std::move(func);
    return *this;
}

CustomOpBuilder& CustomOpBuilder::gradient(GradientFunc func) {
    gradient_ = std::move(func);
    return *this;
}

CustomOp& CustomOpBuilder::build() {
    auto& op = CustomOpRegistry::instance().register_op(schema_);

    for (auto& [backend, kernel] : kernels_) {
        op.register_kernel(backend, std::move(kernel));
    }

    for (auto& [backend, kernel] : multi_kernels_) {
        op.register_kernel_multi(backend, std::move(kernel));
    }

    if (shape_inference_) {
        op.register_shape_inference(std::move(shape_inference_));
    }

    if (type_inference_) {
        op.register_type_inference(std::move(type_inference_));
    }

    if (gradient_) {
        op.register_gradient(std::move(gradient_));
    }

    op.finalize();
    return op;
}

} // namespace custom
} // namespace pyflame_rt
