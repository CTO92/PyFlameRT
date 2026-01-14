#pragma once

#include "pyflame_rt/tensor.hpp"
#include "pyflame_rt/types.hpp"
#include "pyflame_rt/registry.hpp"
#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <any>
#include <unordered_map>
#include <mutex>

namespace pyflame_rt {
namespace custom {

/// Custom operator schema definition
struct OpSchema {
    /// Operator name
    std::string name;

    /// Domain (e.g., "com.mycompany")
    std::string domain = "custom";

    /// Version
    int version = 1;

    /// Input specifications
    struct InputSpec {
        std::string name;
        std::string description;
        DType dtype = DType::Float32;
        bool optional = false;
        std::vector<int64_t> shape_hint;  // Empty for dynamic
    };
    std::vector<InputSpec> inputs;

    /// Output specifications
    struct OutputSpec {
        std::string name;
        std::string description;
        DType dtype = DType::Float32;
    };
    std::vector<OutputSpec> outputs;

    /// Attribute specifications
    struct AttributeSpec {
        std::string name;
        std::string description;
        std::string type;  // "int", "float", "string", "tensor", "ints", etc.
        std::any default_value;
        bool required = false;
    };
    std::vector<AttributeSpec> attributes;

    /// Documentation
    std::string doc;
};

/// Shape inference function type
using ShapeInferenceFunc = std::function<std::vector<TensorInfo>(
    const std::vector<TensorInfo>& inputs,
    const std::unordered_map<std::string, std::any>& attrs)>;

/// Type inference function type
using TypeInferenceFunc = std::function<std::vector<DType>(
    const std::vector<DType>& input_types,
    const std::unordered_map<std::string, std::any>& attrs)>;

/// Kernel function type (same as OpFunction)
using KernelFunc = std::function<Tensor(
    const std::vector<Tensor>& inputs,
    const std::unordered_map<std::string, std::any>& attrs)>;

/// Multi-output kernel function type
using MultiOutputKernelFunc = std::function<std::vector<Tensor>(
    const std::vector<Tensor>& inputs,
    const std::unordered_map<std::string, std::any>& attrs)>;

/// Gradient function type for automatic differentiation
using GradientFunc = std::function<std::vector<Tensor>(
    const std::vector<Tensor>& grad_outputs,
    const std::vector<Tensor>& inputs,
    const std::vector<Tensor>& outputs,
    const std::unordered_map<std::string, std::any>& attrs)>;

/// Backend type for kernel registration
enum class BackendType {
    CPU,
    WSE,
    CUDA,    // Future
    All
};

/// Custom operator definition
class CustomOp {
public:
    explicit CustomOp(const OpSchema& schema);
    ~CustomOp();

    // Non-copyable but movable
    CustomOp(const CustomOp&) = delete;
    CustomOp& operator=(const CustomOp&) = delete;
    CustomOp(CustomOp&&) = default;
    CustomOp& operator=(CustomOp&&) = default;

    // ========================================================================
    // Registration
    // ========================================================================

    /// Register kernel for a backend
    CustomOp& register_kernel(BackendType backend, KernelFunc kernel);

    /// Register multi-output kernel
    CustomOp& register_kernel_multi(BackendType backend,
                                     MultiOutputKernelFunc kernel);

    /// Register shape inference
    CustomOp& register_shape_inference(ShapeInferenceFunc func);

    /// Register type inference
    CustomOp& register_type_inference(TypeInferenceFunc func);

    /// Register gradient function
    CustomOp& register_gradient(GradientFunc func);

    /// Finalize registration
    void finalize();

    // ========================================================================
    // Execution
    // ========================================================================

    /// Execute the operator
    Tensor execute(
        const std::vector<Tensor>& inputs,
        const std::unordered_map<std::string, std::any>& attrs,
        BackendType backend = BackendType::CPU);

    /// Execute with multiple outputs
    std::vector<Tensor> execute_multi(
        const std::vector<Tensor>& inputs,
        const std::unordered_map<std::string, std::any>& attrs,
        BackendType backend = BackendType::CPU);

    // ========================================================================
    // Inference
    // ========================================================================

    /// Infer output shapes
    std::vector<TensorInfo> infer_shapes(
        const std::vector<TensorInfo>& inputs,
        const std::unordered_map<std::string, std::any>& attrs);

    /// Infer output types
    std::vector<DType> infer_types(
        const std::vector<DType>& input_types,
        const std::unordered_map<std::string, std::any>& attrs);

    // ========================================================================
    // Gradient
    // ========================================================================

    /// Compute gradients
    std::vector<Tensor> compute_gradient(
        const std::vector<Tensor>& grad_outputs,
        const std::vector<Tensor>& inputs,
        const std::vector<Tensor>& outputs,
        const std::unordered_map<std::string, std::any>& attrs);

    /// Check if gradient is registered
    bool has_gradient() const;

    // ========================================================================
    // Schema
    // ========================================================================

    /// Get operator schema
    const OpSchema& schema() const { return schema_; }

    /// Get operator name
    const std::string& name() const { return schema_.name; }

    /// Get full name (domain::name)
    std::string full_name() const;

    /// Check if finalized
    bool is_finalized() const { return finalized_; }

private:
    OpSchema schema_;
    std::unordered_map<BackendType, KernelFunc> kernels_;
    std::unordered_map<BackendType, MultiOutputKernelFunc> multi_kernels_;
    ShapeInferenceFunc shape_inference_;
    TypeInferenceFunc type_inference_;
    GradientFunc gradient_;
    bool finalized_ = false;
};

/// Custom operator registry
class CustomOpRegistry {
public:
    static CustomOpRegistry& instance();

    /// Register a custom operator
    CustomOp& register_op(const OpSchema& schema);

    /// Get a custom operator
    CustomOp* get(const std::string& name);
    CustomOp* get(const std::string& domain, const std::string& name);

    /// Check if operator exists
    bool has(const std::string& name) const;
    bool has(const std::string& domain, const std::string& name) const;

    /// List all custom operators
    std::vector<std::string> list() const;

    /// List operators in domain
    std::vector<std::string> list(const std::string& domain) const;

    /// Unregister operator
    void unregister(const std::string& name);
    void unregister(const std::string& domain, const std::string& name);

    /// Clear all custom operators
    void clear();

    /// Get number of registered operators
    size_t size() const;

private:
    CustomOpRegistry() = default;
    ~CustomOpRegistry() = default;

    std::unordered_map<std::string, std::unique_ptr<CustomOp>> ops_;
    mutable std::mutex mutex_;
};

/// Builder for custom operators (fluent API)
class CustomOpBuilder {
public:
    explicit CustomOpBuilder(const std::string& name);

    /// Set domain
    CustomOpBuilder& domain(const std::string& domain);

    /// Set version
    CustomOpBuilder& version(int version);

    /// Add input
    CustomOpBuilder& input(const std::string& name,
                           const std::string& description = "",
                           DType dtype = DType::Float32,
                           bool optional = false);

    /// Add output
    CustomOpBuilder& output(const std::string& name,
                            const std::string& description = "",
                            DType dtype = DType::Float32);

    /// Add int attribute
    CustomOpBuilder& attr_int(const std::string& name,
                               int default_value,
                               const std::string& description = "",
                               bool required = false);

    /// Add float attribute
    CustomOpBuilder& attr_float(const std::string& name,
                                 float default_value,
                                 const std::string& description = "",
                                 bool required = false);

    /// Add string attribute
    CustomOpBuilder& attr_string(const std::string& name,
                                  const std::string& default_value,
                                  const std::string& description = "",
                                  bool required = false);

    /// Add required attribute
    CustomOpBuilder& attr_required(const std::string& name,
                                    const std::string& type,
                                    const std::string& description = "");

    /// Set documentation
    CustomOpBuilder& doc(const std::string& doc);

    /// Set CPU kernel
    CustomOpBuilder& kernel(KernelFunc func);

    /// Set kernel for specific backend
    CustomOpBuilder& kernel(BackendType backend, KernelFunc func);

    /// Set multi-output kernel
    CustomOpBuilder& kernel_multi(MultiOutputKernelFunc func);
    CustomOpBuilder& kernel_multi(BackendType backend, MultiOutputKernelFunc func);

    /// Set shape inference
    CustomOpBuilder& shape_inference(ShapeInferenceFunc func);

    /// Set type inference
    CustomOpBuilder& type_inference(TypeInferenceFunc func);

    /// Set gradient function
    CustomOpBuilder& gradient(GradientFunc func);

    /// Build and register
    CustomOp& build();

private:
    OpSchema schema_;
    std::unordered_map<BackendType, KernelFunc> kernels_;
    std::unordered_map<BackendType, MultiOutputKernelFunc> multi_kernels_;
    ShapeInferenceFunc shape_inference_;
    TypeInferenceFunc type_inference_;
    GradientFunc gradient_;
};

/// Macro for easy operator registration
#define PYFLAME_REGISTER_OP(name) \
    static auto _pyflame_op_##name##_registrar = \
        pyflame_rt::custom::CustomOpBuilder(#name)

/// Macro for registering a simple unary op
#define PYFLAME_REGISTER_UNARY_OP(name, kernel_fn) \
    static struct _pyflame_##name##_registrar { \
        _pyflame_##name##_registrar() { \
            pyflame_rt::custom::CustomOpBuilder(#name) \
                .input("X", "Input tensor") \
                .output("Y", "Output tensor") \
                .kernel(kernel_fn) \
                .build(); \
        } \
    } _pyflame_##name##_instance

/// Macro for registering a simple binary op
#define PYFLAME_REGISTER_BINARY_OP(name, kernel_fn) \
    static struct _pyflame_##name##_registrar { \
        _pyflame_##name##_registrar() { \
            pyflame_rt::custom::CustomOpBuilder(#name) \
                .input("A", "First input tensor") \
                .input("B", "Second input tensor") \
                .output("C", "Output tensor") \
                .kernel(kernel_fn) \
                .build(); \
        } \
    } _pyflame_##name##_instance

} // namespace custom
} // namespace pyflame_rt
