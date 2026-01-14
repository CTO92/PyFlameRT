# Extending PyFlameRT

This guide explains how to extend PyFlameRT with custom operators, backends, and model formats.

---

## Table of Contents

1. [Custom Operators](#custom-operators)
   - [Operator Basics](#operator-basics)
   - [Implementing a Custom Operator](#implementing-a-custom-operator)
   - [Registering Operators](#registering-operators)
   - [Handling Attributes](#handling-attributes)
   - [Multi-Output Operators](#multi-output-operators)
   - [Best Practices](#operator-best-practices)
2. [Custom Backends](#custom-backends)
   - [Backend Architecture](#backend-architecture)
   - [Implementing a Backend](#implementing-a-backend)
   - [Memory Management](#memory-management)
   - [Backend Registration](#backend-registration)
3. [Custom Model Formats](#custom-model-formats)
   - [Model I/O Interface](#model-io-interface)
   - [Implementing a Loader](#implementing-a-loader)
4. [Extending Serving Infrastructure](#extending-serving-infrastructure)
   - [Custom HTTP Handlers](#custom-http-handlers)
   - [Custom Metrics](#custom-metrics)
   - [Pre/Post Processing Hooks](#prepost-processing-hooks)
5. [Advanced Topics](#advanced-topics)
   - [Operator Fusion](#operator-fusion)
   - [Graph Optimization](#graph-optimization)
   - [Multi-Threading](#multi-threading)

---

## Custom Operators

### Operator Basics

PyFlameRT operators are functions that:
1. Take a vector of input tensors
2. Take a map of attributes
3. Return an output tensor (or tensors)

The operator signature:

```cpp
using OpFunction = std::function<Tensor(
    const std::vector<Tensor>& inputs,
    const std::unordered_map<std::string, std::any>& attributes)>;
```

### Implementing a Custom Operator

#### Step 1: Create the Operator Source File

Create a new file in `src/backends/cpu/ops/`:

```cpp
// src/backends/cpu/ops/custom_ops.cpp
#include <pyflame_rt/tensor.hpp>
#include <pyflame_rt/registry.hpp>
#include <pyflame_rt/errors.hpp>
#include <cmath>

namespace pyflame_rt {
namespace cpu {

// Simple unary operator: Square
Tensor cpu_square(
    const std::vector<Tensor>& inputs,
    const std::unordered_map<std::string, std::any>& /* attrs */
) {
    if (inputs.empty()) {
        throw ValidationError("Square requires 1 input");
    }

    const Tensor& input = inputs[0];
    Tensor output(input.shape(), input.dtype());

    if (input.dtype() == DType::Float32) {
        const float* in_data = input.data_ptr<float>();
        float* out_data = output.data_ptr<float>();

        for (int64_t i = 0; i < input.num_elements(); ++i) {
            out_data[i] = in_data[i] * in_data[i];
        }
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_data = input.data_ptr<double>();
        double* out_data = output.data_ptr<double>();

        for (int64_t i = 0; i < input.num_elements(); ++i) {
            out_data[i] = in_data[i] * in_data[i];
        }
    }
    else {
        throw ValidationError("Square: unsupported dtype");
    }

    return output;
}

// Binary operator with broadcasting: SafeDiv (handles division by zero)
Tensor cpu_safe_div(
    const std::vector<Tensor>& inputs,
    const std::unordered_map<std::string, std::any>& attrs
) {
    if (inputs.size() < 2) {
        throw ValidationError("SafeDiv requires 2 inputs");
    }

    const Tensor& a = inputs[0];
    const Tensor& b = inputs[1];

    // Get epsilon from attributes (default: 1e-7)
    float epsilon = 1e-7f;
    if (attrs.count("epsilon")) {
        epsilon = std::any_cast<float>(attrs.at("epsilon"));
    }

    // For simplicity, require same shapes (no broadcasting in this example)
    if (a.shape() != b.shape()) {
        throw ValidationError("SafeDiv: shapes must match");
    }

    Tensor output(a.shape(), a.dtype());

    if (a.dtype() == DType::Float32) {
        const float* a_data = a.data_ptr<float>();
        const float* b_data = b.data_ptr<float>();
        float* out_data = output.data_ptr<float>();

        for (int64_t i = 0; i < a.num_elements(); ++i) {
            float divisor = b_data[i];
            if (std::abs(divisor) < epsilon) {
                divisor = (divisor >= 0) ? epsilon : -epsilon;
            }
            out_data[i] = a_data[i] / divisor;
        }
    }
    else {
        throw ValidationError("SafeDiv: only Float32 supported");
    }

    return output;
}

} // namespace cpu
} // namespace pyflame_rt
```

#### Step 2: Register the Operators

Add registration at static initialization:

```cpp
// Continue in custom_ops.cpp

namespace pyflame_rt {
namespace cpu {

struct CustomOpsRegistrar {
    CustomOpsRegistrar() {
        auto& reg = OperatorRegistry::instance();
        reg.register_op("Square", cpu_square);
        reg.register_op("SafeDiv", cpu_safe_div);
    }
};

// Static instance triggers registration at load time
static CustomOpsRegistrar custom_ops_registrar;

} // namespace cpu
} // namespace pyflame_rt
```

#### Step 3: Add to Build System

Update `src/backends/cpu/CMakeLists.txt` (or equivalent):

```cmake
target_sources(pyflame_rt PRIVATE
    ops/math.cpp
    ops/activation.cpp
    ops/tensor_ops.cpp
    ops/reduction.cpp
    ops/nn.cpp
    ops/custom_ops.cpp  # Add new file
)
```

### Handling Attributes

Operators can receive attributes of various types:

```cpp
Tensor cpu_my_op(
    const std::vector<Tensor>& inputs,
    const std::unordered_map<std::string, std::any>& attrs
) {
    // Integer attribute
    int64_t axis = 0;
    if (attrs.count("axis")) {
        axis = std::any_cast<int64_t>(attrs.at("axis"));
    }

    // Float attribute
    float alpha = 1.0f;
    if (attrs.count("alpha")) {
        alpha = std::any_cast<float>(attrs.at("alpha"));
    }

    // String attribute
    std::string mode = "nearest";
    if (attrs.count("mode")) {
        mode = std::any_cast<std::string>(attrs.at("mode"));
    }

    // Integer array attribute
    std::vector<int64_t> kernel_shape = {3, 3};
    if (attrs.count("kernel_shape")) {
        kernel_shape = std::any_cast<std::vector<int64_t>>(attrs.at("kernel_shape"));
    }

    // Float array attribute
    std::vector<float> scales;
    if (attrs.count("scales")) {
        scales = std::any_cast<std::vector<float>>(attrs.at("scales"));
    }

    // ... implement operation
}
```

### Multi-Output Operators

For operators that produce multiple outputs, return the first output and use the registry's multi-output variant:

```cpp
// Multi-output operator signature
using MultiOutputOpFunction = std::function<std::vector<Tensor>(
    const std::vector<Tensor>& inputs,
    const std::unordered_map<std::string, std::any>& attributes)>;

// Example: Split operator
std::vector<Tensor> cpu_split(
    const std::vector<Tensor>& inputs,
    const std::unordered_map<std::string, std::any>& attrs
) {
    const Tensor& input = inputs[0];
    int64_t axis = std::any_cast<int64_t>(attrs.at("axis"));
    auto split_sizes = std::any_cast<std::vector<int64_t>>(attrs.at("split"));

    std::vector<Tensor> outputs;

    // ... implement split logic

    return outputs;
}
```

### Operator Best Practices

1. **Validate inputs early:**
   ```cpp
   if (inputs.empty()) {
       throw ValidationError("OpName requires at least 1 input");
   }
   ```

2. **Support common dtypes:**
   ```cpp
   switch (input.dtype()) {
       case DType::Float32: /* ... */ break;
       case DType::Float64: /* ... */ break;
       case DType::Int64:   /* ... */ break;
       default:
           throw ValidationError("OpName: unsupported dtype");
   }
   ```

3. **Use template helpers for dtype dispatch:**
   ```cpp
   template<typename T>
   void compute_op(const T* input, T* output, int64_t n) {
       for (int64_t i = 0; i < n; ++i) {
           output[i] = /* computation */;
       }
   }

   Tensor cpu_my_op(const std::vector<Tensor>& inputs, ...) {
       const Tensor& input = inputs[0];
       Tensor output(input.shape(), input.dtype());

       if (input.dtype() == DType::Float32) {
           compute_op(input.data_ptr<float>(),
                     output.data_ptr<float>(),
                     input.num_elements());
       }
       // ... other dtypes
   }
   ```

4. **Handle broadcasting correctly:**
   ```cpp
   std::vector<int64_t> broadcast_shapes(
       const std::vector<int64_t>& a,
       const std::vector<int64_t>& b
   ) {
       std::vector<int64_t> result;
       auto ai = a.rbegin();
       auto bi = b.rbegin();

       while (ai != a.rend() || bi != b.rend()) {
           int64_t da = (ai != a.rend()) ? *ai++ : 1;
           int64_t db = (bi != b.rend()) ? *bi++ : 1;

           if (da != db && da != 1 && db != 1) {
               throw ValidationError("Shapes not broadcastable");
           }
           result.push_back(std::max(da, db));
       }
       std::reverse(result.begin(), result.end());
       return result;
   }
   ```

5. **Document operator semantics:**
   ```cpp
   /**
    * @brief Compute element-wise square of input tensor.
    *
    * @param inputs Single input tensor
    * @param attrs (none required)
    * @return Output tensor with same shape and dtype
    *
    * Supports: Float32, Float64, Int32, Int64
    */
   Tensor cpu_square(const std::vector<Tensor>& inputs, ...);
   ```

---

## Custom Backends

### Backend Architecture

A backend in PyFlameRT:
1. Executes graphs on a specific hardware target
2. Manages memory allocation for that target
3. Implements operators optimized for the target

```
              +-----------------------+
              |   InferenceSession    |
              +-----------------------+
                        |
                        v
              +-----------------------+
              |    Backend Manager    |
              +-----------------------+
               /         |           \
              v          v            v
         +-------+  +--------+  +-----------+
         | CPU   |  |  WSE   |  |  Custom   |
         |Backend|  |Backend |  |  Backend  |
         +-------+  +--------+  +-----------+
```

### Implementing a Backend

#### Step 1: Create Backend Header

```cpp
// include/pyflame_rt/backends/my_backend.hpp
#pragma once

#include <pyflame_rt/backend.hpp>

namespace pyflame_rt {

class MyBackend : public Backend {
public:
    MyBackend();
    ~MyBackend() override;

    // Backend info
    std::string name() const override { return "MyBackend"; }
    std::string device_type() const override { return "my_device"; }

    // Initialization
    bool initialize(const SessionOptions& options) override;
    void shutdown() override;

    // Execution
    std::vector<Tensor> execute(
        const Graph& graph,
        const std::unordered_map<std::string, Tensor>& inputs) override;

    // Memory management
    Tensor allocate(const std::vector<int64_t>& shape, DType dtype) override;
    void deallocate(Tensor& tensor) override;

    // Capabilities
    bool supports_op(const std::string& op_type) const override;
    std::vector<std::string> supported_ops() const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace pyflame_rt
```

#### Step 2: Implement the Backend

```cpp
// src/backends/my_backend/my_backend.cpp
#include <pyflame_rt/backends/my_backend.hpp>
#include <pyflame_rt/graph.hpp>
#include <pyflame_rt/errors.hpp>

namespace pyflame_rt {

struct MyBackend::Impl {
    bool initialized = false;
    SessionOptions options;

    // Device-specific state
    // void* device_context = nullptr;

    // Operator implementations for this backend
    std::unordered_map<std::string, std::function<Tensor(
        const std::vector<Tensor>&,
        const std::unordered_map<std::string, std::any>&)>> ops;

    void register_ops() {
        ops["Add"] = [this](auto& inputs, auto& attrs) {
            return execute_add(inputs, attrs);
        };
        ops["MatMul"] = [this](auto& inputs, auto& attrs) {
            return execute_matmul(inputs, attrs);
        };
        // Register more operators...
    }

    Tensor execute_add(
        const std::vector<Tensor>& inputs,
        const std::unordered_map<std::string, std::any>& attrs
    ) {
        // Device-optimized implementation
        // ...
    }

    Tensor execute_matmul(
        const std::vector<Tensor>& inputs,
        const std::unordered_map<std::string, std::any>& attrs
    ) {
        // Device-optimized implementation
        // ...
    }
};

MyBackend::MyBackend() : impl_(std::make_unique<Impl>()) {
    impl_->register_ops();
}

MyBackend::~MyBackend() {
    if (impl_->initialized) {
        shutdown();
    }
}

bool MyBackend::initialize(const SessionOptions& options) {
    impl_->options = options;

    // Initialize device
    // impl_->device_context = my_device_init();
    // if (!impl_->device_context) return false;

    impl_->initialized = true;
    return true;
}

void MyBackend::shutdown() {
    if (impl_->initialized) {
        // Clean up device resources
        // my_device_shutdown(impl_->device_context);
        impl_->initialized = false;
    }
}

std::vector<Tensor> MyBackend::execute(
    const Graph& graph,
    const std::unordered_map<std::string, Tensor>& inputs
) {
    if (!impl_->initialized) {
        throw PyFlameRTError("Backend not initialized");
    }

    // Get execution order
    auto sorted_nodes = graph.topological_sort();

    // Tensor storage for intermediate values
    std::unordered_map<std::string, Tensor> tensors;

    // Copy inputs
    for (const auto& [name, tensor] : inputs) {
        tensors[name] = tensor;
    }

    // Copy initializers
    for (const auto& [name, tensor] : graph.initializers()) {
        tensors[name] = tensor;
    }

    // Execute nodes
    for (const auto& node : sorted_nodes) {
        // Gather inputs
        std::vector<Tensor> node_inputs;
        for (const auto& input_name : node->inputs()) {
            if (tensors.count(input_name)) {
                node_inputs.push_back(tensors.at(input_name));
            }
        }

        // Get operator
        if (!impl_->ops.count(node->op_type())) {
            throw UnsupportedOperatorError(node->op_type());
        }

        // Prepare attributes
        std::unordered_map<std::string, std::any> attrs;
        for (const auto& attr_name : node->attribute_names()) {
            // Copy attributes from node...
        }

        // Execute
        Tensor output = impl_->ops[node->op_type()](node_inputs, attrs);

        // Store output
        if (!node->outputs().empty()) {
            tensors[node->outputs()[0]] = std::move(output);
        }
    }

    // Gather outputs
    std::vector<Tensor> outputs;
    for (const auto& output_info : graph.outputs()) {
        outputs.push_back(tensors.at(output_info.name));
    }

    return outputs;
}

Tensor MyBackend::allocate(const std::vector<int64_t>& shape, DType dtype) {
    // Allocate on device
    // For example, using device-specific allocator:
    // void* device_ptr = my_device_alloc(size);
    // return Tensor(device_ptr, shape, dtype);

    // Default: use CPU allocation
    return Tensor(shape, dtype);
}

void MyBackend::deallocate(Tensor& tensor) {
    // Free device memory if needed
    // my_device_free(tensor.data());
}

bool MyBackend::supports_op(const std::string& op_type) const {
    return impl_->ops.count(op_type) > 0;
}

std::vector<std::string> MyBackend::supported_ops() const {
    std::vector<std::string> ops;
    for (const auto& [name, _] : impl_->ops) {
        ops.push_back(name);
    }
    return ops;
}

} // namespace pyflame_rt
```

### Memory Management

For hardware backends with separate memory:

```cpp
class DeviceMemoryManager {
public:
    // Allocate device memory
    void* allocate(size_t bytes) {
        void* ptr = device_malloc(bytes);
        allocations_[ptr] = bytes;
        total_allocated_ += bytes;
        return ptr;
    }

    // Free device memory
    void deallocate(void* ptr) {
        if (allocations_.count(ptr)) {
            total_allocated_ -= allocations_[ptr];
            allocations_.erase(ptr);
            device_free(ptr);
        }
    }

    // Transfer host -> device
    void copy_to_device(void* device_ptr, const void* host_ptr, size_t bytes) {
        device_memcpy_h2d(device_ptr, host_ptr, bytes);
    }

    // Transfer device -> host
    void copy_to_host(void* host_ptr, const void* device_ptr, size_t bytes) {
        device_memcpy_d2h(host_ptr, device_ptr, bytes);
    }

    // Memory pool for efficient allocation
    void* allocate_from_pool(size_t bytes) {
        // Check free list first
        for (auto it = free_list_.begin(); it != free_list_.end(); ++it) {
            if (it->second >= bytes) {
                void* ptr = it->first;
                free_list_.erase(it);
                return ptr;
            }
        }
        return allocate(bytes);
    }

    void return_to_pool(void* ptr, size_t bytes) {
        free_list_[ptr] = bytes;
    }

private:
    std::unordered_map<void*, size_t> allocations_;
    std::map<void*, size_t> free_list_;
    size_t total_allocated_ = 0;
};
```

### Backend Registration

Register backends for automatic discovery:

```cpp
// src/backends/backend_registry.cpp

#include <pyflame_rt/backend.hpp>
#include <memory>
#include <unordered_map>

namespace pyflame_rt {

class BackendRegistry {
public:
    static BackendRegistry& instance() {
        static BackendRegistry registry;
        return registry;
    }

    using BackendFactory = std::function<std::unique_ptr<Backend>()>;

    void register_backend(const std::string& name, BackendFactory factory) {
        factories_[name] = std::move(factory);
    }

    std::unique_ptr<Backend> create(const std::string& name) {
        if (factories_.count(name)) {
            return factories_[name]();
        }
        return nullptr;
    }

    std::vector<std::string> available_backends() const {
        std::vector<std::string> names;
        for (const auto& [name, _] : factories_) {
            names.push_back(name);
        }
        return names;
    }

private:
    std::unordered_map<std::string, BackendFactory> factories_;
};

// Register CPU backend
struct CPUBackendRegistrar {
    CPUBackendRegistrar() {
        BackendRegistry::instance().register_backend(
            "CPUExecutionProvider",
            []() { return std::make_unique<CPUBackend>(); }
        );
    }
};
static CPUBackendRegistrar cpu_registrar;

// Register custom backend
struct MyBackendRegistrar {
    MyBackendRegistrar() {
        BackendRegistry::instance().register_backend(
            "MyExecutionProvider",
            []() { return std::make_unique<MyBackend>(); }
        );
    }
};
static MyBackendRegistrar my_registrar;

} // namespace pyflame_rt
```

---

## Custom Model Formats

### Model I/O Interface

```cpp
// include/pyflame_rt/io/model_loader.hpp
#pragma once

#include <pyflame_rt/graph.hpp>
#include <memory>
#include <string>

namespace pyflame_rt {
namespace io {

class ModelLoader {
public:
    virtual ~ModelLoader() = default;

    // Check if this loader can handle the file
    virtual bool can_load(const std::string& path) const = 0;

    // Load model from file
    virtual std::shared_ptr<Graph> load(const std::string& path) const = 0;

    // Get supported extensions
    virtual std::vector<std::string> extensions() const = 0;
};

class ModelSaver {
public:
    virtual ~ModelSaver() = default;

    // Save model to file
    virtual void save(const Graph& graph, const std::string& path) const = 0;

    // Get default extension
    virtual std::string default_extension() const = 0;
};

} // namespace io
} // namespace pyflame_rt
```

### Implementing a Loader

Example: JSON model format loader

```cpp
// src/io/json_loader.cpp

#include <pyflame_rt/io/model_loader.hpp>
#include <pyflame_rt/errors.hpp>
#include <fstream>
#include <nlohmann/json.hpp>  // Using nlohmann/json library

namespace pyflame_rt {
namespace io {

using json = nlohmann::json;

class JsonModelLoader : public ModelLoader {
public:
    bool can_load(const std::string& path) const override {
        return path.size() > 5 &&
               path.substr(path.size() - 5) == ".json";
    }

    std::shared_ptr<Graph> load(const std::string& path) const override {
        std::ifstream file(path);
        if (!file) {
            throw InvalidModelError("Cannot open file: " + path);
        }

        json j;
        try {
            file >> j;
        } catch (const json::parse_error& e) {
            throw InvalidModelError("JSON parse error: " + std::string(e.what()));
        }

        auto graph = std::make_shared<Graph>(
            j.value("name", "model")
        );

        // Load metadata
        if (j.contains("metadata")) {
            auto& meta = graph->metadata();
            auto& jmeta = j["metadata"];
            meta.producer_name = jmeta.value("producer", "");
            meta.producer_version = jmeta.value("version", "");
            meta.doc_string = jmeta.value("description", "");
        }

        // Load inputs
        if (j.contains("inputs")) {
            for (const auto& jinput : j["inputs"]) {
                TensorInfo info;
                info.name = jinput["name"];
                info.dtype = string_to_dtype(jinput["dtype"]);
                for (const auto& dim : jinput["shape"]) {
                    if (dim.is_null()) {
                        info.shape.push_back(std::nullopt);
                    } else {
                        info.shape.push_back(dim.get<int64_t>());
                    }
                }
                graph->add_input(info);
            }
        }

        // Load nodes
        if (j.contains("nodes")) {
            for (const auto& jnode : j["nodes"]) {
                auto node = std::make_shared<Node>(
                    jnode["op_type"],
                    jnode["name"],
                    jnode["inputs"].get<std::vector<std::string>>(),
                    jnode["outputs"].get<std::vector<std::string>>()
                );

                // Load attributes
                if (jnode.contains("attributes")) {
                    for (auto& [key, val] : jnode["attributes"].items()) {
                        if (val.is_number_integer()) {
                            node->set_attribute(key, val.get<int64_t>());
                        } else if (val.is_number_float()) {
                            node->set_attribute(key, val.get<double>());
                        } else if (val.is_string()) {
                            node->set_attribute(key, val.get<std::string>());
                        } else if (val.is_array()) {
                            if (val[0].is_number_integer()) {
                                node->set_attribute(key,
                                    val.get<std::vector<int64_t>>());
                            } else {
                                node->set_attribute(key,
                                    val.get<std::vector<double>>());
                            }
                        }
                    }
                }

                graph->add_node(node);
            }
        }

        // Load outputs
        if (j.contains("outputs")) {
            for (const auto& joutput : j["outputs"]) {
                TensorInfo info;
                info.name = joutput["name"];
                info.dtype = string_to_dtype(joutput.value("dtype", "float32"));
                graph->add_output(info);
            }
        }

        // Load initializers (weights)
        if (j.contains("initializers")) {
            for (const auto& jinit : j["initializers"]) {
                std::string name = jinit["name"];
                auto shape = jinit["shape"].get<std::vector<int64_t>>();
                DType dtype = string_to_dtype(jinit["dtype"]);

                Tensor tensor(shape, dtype);

                // Load data (base64 encoded or inline array)
                if (jinit.contains("data_base64")) {
                    // Decode base64 data
                    std::string encoded = jinit["data_base64"];
                    // ... decode and copy to tensor
                } else if (jinit.contains("data")) {
                    // Inline float array
                    auto data = jinit["data"].get<std::vector<float>>();
                    std::memcpy(tensor.data(), data.data(),
                               data.size() * sizeof(float));
                }

                graph->add_initializer(name, std::move(tensor));
            }
        }

        return graph;
    }

    std::vector<std::string> extensions() const override {
        return {".json"};
    }
};

// Register loader
struct JsonLoaderRegistrar {
    JsonLoaderRegistrar() {
        // Register with model I/O system
    }
};
static JsonLoaderRegistrar json_registrar;

} // namespace io
} // namespace pyflame_rt
```

---

## Extending Serving Infrastructure

The serving module can be extended with custom HTTP handlers, metrics, and pre/post processing hooks.

### Custom HTTP Handlers

Add custom endpoints to the model server:

```cpp
#include "pyflame_rt/serving/http_server.hpp"

namespace pyflame_rt {
namespace serving {

class CustomModelServer : public ModelServer {
public:
    CustomModelServer(const ServerConfig& config)
        : ModelServer(config)
    {
        setup_custom_routes();
    }

private:
    void setup_custom_routes() {
        // Add a custom health endpoint with detailed checks
        http_server_->route("GET", "/health/detailed",
            [this](const HTTPRequest& req) {
                return handle_detailed_health(req);
            });

        // Add a custom preprocessing endpoint
        http_server_->route("POST", "/v1/preprocess",
            [this](const HTTPRequest& req) {
                return handle_preprocess(req);
            });

        // Add a custom batch endpoint
        http_server_->route("POST", "/v1/batch",
            [this](const HTTPRequest& req) {
                return handle_batch_request(req);
            });
    }

    HTTPResponse handle_detailed_health(const HTTPRequest& req) {
        std::ostringstream json;
        json << "{\n";
        json << "  \"server\": \"running\",\n";
        json << "  \"models\": {\n";

        auto models = registry().list_models();
        for (size_t i = 0; i < models.size(); ++i) {
            auto instance = registry().get(models[i]);
            auto stats = instance->get_stats();

            json << "    \"" << models[i] << "\": {\n";
            json << "      \"ready\": " << (instance->is_ready() ? "true" : "false") << ",\n";
            json << "      \"requests\": " << stats.total_requests << ",\n";
            json << "      \"avg_latency_ms\": " << stats.avg_latency_ms << "\n";
            json << "    }";
            if (i + 1 < models.size()) json << ",";
            json << "\n";
        }

        json << "  }\n";
        json << "}";

        return HTTPResponse::json(json.str());
    }

    HTTPResponse handle_preprocess(const HTTPRequest& req) {
        // Custom preprocessing logic
        // Parse request, transform data, return preprocessed result
        return HTTPResponse::json("{\"status\": \"preprocessed\"}");
    }

    HTTPResponse handle_batch_request(const HTTPRequest& req) {
        // Custom batch handling logic
        return HTTPResponse::json("{\"status\": \"batch_processed\"}");
    }
};

} // namespace serving
} // namespace pyflame_rt
```

### Custom Metrics

Add custom metrics to track application-specific data:

```cpp
#include "pyflame_rt/serving/metrics.hpp"

namespace pyflame_rt {
namespace serving {

// Register custom metrics
void register_custom_metrics() {
    auto& registry = MetricsRegistry::instance();

    // Custom counter for specific use case
    registry.counter_inc("myapp_image_classifications_total",
        {{"class", "cat"}, {"confidence", "high"}});

    // Custom histogram for preprocessing time
    registry.histogram_observe("myapp_preprocess_seconds",
        0.015,  // 15ms preprocessing
        {{"model", "resnet50"}});

    // Custom gauge for queue depth by priority
    registry.gauge_set("myapp_priority_queue_depth",
        42.0,
        {{"priority", "high"}});
}

// Custom metrics collector
class CustomMetricsCollector {
public:
    void record_classification(const std::string& class_name,
                                double confidence) {
        std::string confidence_bucket =
            confidence > 0.9 ? "high" :
            confidence > 0.5 ? "medium" : "low";

        MetricsRegistry::instance().counter_inc(
            "myapp_image_classifications_total",
            {{"class", class_name}, {"confidence", confidence_bucket}});
    }

    void record_preprocessing_time(const std::string& model,
                                    double seconds) {
        MetricsRegistry::instance().histogram_observe(
            "myapp_preprocess_seconds",
            seconds,
            {{"model", model}});
    }

    void update_queue_depth(const std::string& priority,
                            size_t depth) {
        MetricsRegistry::instance().gauge_set(
            "myapp_priority_queue_depth",
            static_cast<double>(depth),
            {{"priority", priority}});
    }
};

} // namespace serving
} // namespace pyflame_rt
```

Python integration for custom metrics:

```python
from pyflame_rt.serving.metrics import MetricsRegistry

# Get the registry
registry = MetricsRegistry.instance()

# Export including custom metrics
metrics = registry.export_prometheus()
print(metrics)
```

### Pre/Post Processing Hooks

Add custom processing before and after inference:

```cpp
#include "pyflame_rt/serving/model_registry.hpp"

namespace pyflame_rt {
namespace serving {

// Custom model instance with pre/post processing
class CustomModelInstance {
public:
    using PreprocessFunc = std::function<InferRequest(const InferRequest&)>;
    using PostprocessFunc = std::function<InferResponse(const InferResponse&)>;

    CustomModelInstance(std::shared_ptr<ModelInstance> base)
        : base_instance_(base)
    {}

    void set_preprocessor(PreprocessFunc func) {
        preprocess_ = std::move(func);
    }

    void set_postprocessor(PostprocessFunc func) {
        postprocess_ = std::move(func);
    }

    InferResponse infer(const InferRequest& request) {
        // Preprocess
        InferRequest processed_request = request;
        if (preprocess_) {
            processed_request = preprocess_(request);
        }

        // Run inference
        InferResponse response = base_instance_->infer(processed_request);

        // Postprocess
        if (postprocess_ && response.success) {
            response = postprocess_(response);
        }

        return response;
    }

private:
    std::shared_ptr<ModelInstance> base_instance_;
    PreprocessFunc preprocess_;
    PostprocessFunc postprocess_;
};

// Example usage
void setup_image_classification_hooks(CustomModelInstance& instance) {
    // Preprocessing: Normalize input
    instance.set_preprocessor([](const InferRequest& req) {
        InferRequest processed = req;

        // Example: Normalize image input to [0, 1]
        if (req.inputs.count("input") > 0) {
            const Tensor& input = req.inputs.at("input");
            Tensor normalized(input.shape(), input.dtype());

            const float* in_data = input.data_ptr<float>();
            float* out_data = normalized.data_ptr<float>();

            for (int64_t i = 0; i < input.num_elements(); ++i) {
                out_data[i] = in_data[i] / 255.0f;
            }

            processed.inputs["input"] = std::move(normalized);
        }

        return processed;
    });

    // Postprocessing: Apply softmax to logits
    instance.set_postprocessor([](const InferResponse& resp) {
        InferResponse processed = resp;

        if (resp.outputs.count("logits") > 0) {
            const Tensor& logits = resp.outputs.at("logits");
            Tensor probs(logits.shape(), logits.dtype());

            const float* in_data = logits.data_ptr<float>();
            float* out_data = probs.data_ptr<float>();

            // Compute softmax
            int64_t num_classes = logits.shape().back();
            float max_val = *std::max_element(in_data, in_data + num_classes);
            float sum = 0.0f;

            for (int64_t i = 0; i < num_classes; ++i) {
                out_data[i] = std::exp(in_data[i] - max_val);
                sum += out_data[i];
            }
            for (int64_t i = 0; i < num_classes; ++i) {
                out_data[i] /= sum;
            }

            processed.outputs["probabilities"] = std::move(probs);
        }

        return processed;
    });
}

} // namespace serving
} // namespace pyflame_rt
```

Python hooks for inference:

```python
import numpy as np
from pyflame_rt.serving import ModelClient, InferenceRequest

class ImageClassificationClient(ModelClient):
    """Client with built-in preprocessing and postprocessing."""

    def __init__(self, url, mean=None, std=None, class_names=None):
        super().__init__(url)
        self.mean = mean or np.array([0.485, 0.456, 0.406])
        self.std = std or np.array([0.229, 0.224, 0.225])
        self.class_names = class_names

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Normalize image using ImageNet statistics."""
        # Convert to float and normalize
        img = image.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std

        # NHWC -> NCHW
        if img.ndim == 3:
            img = img.transpose(2, 0, 1)
        elif img.ndim == 4:
            img = img.transpose(0, 3, 1, 2)

        # Add batch dimension if needed
        if img.ndim == 3:
            img = img[np.newaxis, ...]

        return img

    def postprocess(self, logits: np.ndarray) -> dict:
        """Convert logits to class probabilities."""
        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        # Get top-k predictions
        top_k = 5
        top_indices = np.argsort(probs[0])[-top_k:][::-1]

        results = []
        for idx in top_indices:
            name = self.class_names[idx] if self.class_names else f"class_{idx}"
            results.append({
                "class_id": int(idx),
                "class_name": name,
                "probability": float(probs[0, idx])
            })

        return {
            "predictions": results,
            "probabilities": probs[0].tolist()
        }

    def classify(self, image: np.ndarray) -> dict:
        """Full pipeline: preprocess -> infer -> postprocess."""
        # Preprocess
        input_tensor = self.preprocess(image)

        # Infer
        response = self.infer(
            model="image_classifier",
            inputs={"input": input_tensor}
        )

        if not response.success:
            return {"error": response.error_message}

        # Postprocess
        logits = response.outputs.get("logits") or response.outputs.get("output")
        return self.postprocess(logits)

# Usage
client = ImageClassificationClient(
    "http://localhost:8080",
    class_names=["cat", "dog", "bird", ...]  # ImageNet class names
)

image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
result = client.classify(image)
print(result["predictions"][:3])
```

---

## Advanced Topics

### Operator Fusion

Operator fusion combines multiple operations into one for better performance:

```cpp
// Example: Fuse Conv + BatchNorm + ReLU
class ConvBnReluFusion {
public:
    bool can_fuse(const Node& conv, const Node& bn, const Node& relu) const {
        return conv.op_type() == "Conv" &&
               bn.op_type() == "BatchNormalization" &&
               relu.op_type() == "Relu" &&
               conv.outputs()[0] == bn.inputs()[0] &&
               bn.outputs()[0] == relu.inputs()[0];
    }

    std::shared_ptr<Node> fuse(
        const Node& conv,
        const Node& bn,
        const Node& relu,
        Graph& graph
    ) {
        // Create fused node
        auto fused = std::make_shared<Node>(
            "FusedConvBnRelu",
            conv.name() + "_fused",
            conv.inputs(),
            relu.outputs()
        );

        // Copy conv attributes
        for (const auto& attr : conv.attribute_names()) {
            // Copy attribute...
        }

        // Fold BatchNorm into conv weights
        const Tensor& weight = graph.get_initializer(conv.inputs()[1]);
        const Tensor& scale = graph.get_initializer(bn.inputs()[1]);
        const Tensor& bias = graph.get_initializer(bn.inputs()[2]);
        const Tensor& mean = graph.get_initializer(bn.inputs()[3]);
        const Tensor& var = graph.get_initializer(bn.inputs()[4]);

        // Compute fused weights: w_fused = w * scale / sqrt(var + eps)
        // Compute fused bias: b_fused = (b - mean) * scale / sqrt(var + eps) + bias
        Tensor fused_weight = fold_bn_weights(weight, scale, var);
        Tensor fused_bias = fold_bn_bias(weight, scale, bias, mean, var);

        graph.add_initializer(conv.inputs()[1] + "_fused", std::move(fused_weight));
        graph.add_initializer(conv.name() + "_fused_bias", std::move(fused_bias));

        return fused;
    }
};
```

### Graph Optimization

Implement graph optimization passes:

```cpp
class GraphOptimizer {
public:
    void add_pass(std::shared_ptr<OptimizationPass> pass) {
        passes_.push_back(std::move(pass));
    }

    void optimize(Graph& graph, int optimization_level) {
        for (auto& pass : passes_) {
            if (pass->level() <= optimization_level) {
                pass->run(graph);
            }
        }
    }

private:
    std::vector<std::shared_ptr<OptimizationPass>> passes_;
};

class OptimizationPass {
public:
    virtual ~OptimizationPass() = default;
    virtual void run(Graph& graph) = 0;
    virtual int level() const = 0;
    virtual std::string name() const = 0;
};

// Example: Constant folding pass
class ConstantFoldingPass : public OptimizationPass {
public:
    void run(Graph& graph) override {
        // Find nodes with all constant inputs
        // Evaluate them at compile time
        // Replace with initializers
    }

    int level() const override { return 1; }
    std::string name() const override { return "ConstantFolding"; }
};

// Example: Dead code elimination
class DeadCodeEliminationPass : public OptimizationPass {
public:
    void run(Graph& graph) override {
        // Find nodes whose outputs are never used
        // Remove them from the graph
    }

    int level() const override { return 1; }
    std::string name() const override { return "DeadCodeElimination"; }
};
```

### Multi-Threading

Implement parallel execution for operators:

```cpp
#include <thread>
#include <future>

class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads) {
        for (size_t i = 0; i < num_threads; ++i) {
            workers_.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(mutex_);
                        cv_.wait(lock, [this] {
                            return stop_ || !tasks_.empty();
                        });
                        if (stop_ && tasks_.empty()) return;
                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }
                    task();
                }
            });
        }
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& worker : workers_) {
            worker.join();
        }
    }

    template<typename F>
    auto submit(F&& f) -> std::future<decltype(f())> {
        using ReturnType = decltype(f());
        auto task = std::make_shared<std::packaged_task<ReturnType()>>(
            std::forward<F>(f)
        );
        auto result = task->get_future();
        {
            std::unique_lock<std::mutex> lock(mutex_);
            tasks_.emplace([task]() { (*task)(); });
        }
        cv_.notify_one();
        return result;
    }

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_ = false;
};

// Use in operator implementation
Tensor cpu_parallel_add(
    const std::vector<Tensor>& inputs,
    const std::unordered_map<std::string, std::any>& attrs
) {
    static ThreadPool pool(std::thread::hardware_concurrency());

    const Tensor& a = inputs[0];
    const Tensor& b = inputs[1];
    Tensor output(a.shape(), a.dtype());

    const float* a_data = a.data_ptr<float>();
    const float* b_data = b.data_ptr<float>();
    float* out_data = output.data_ptr<float>();

    int64_t n = a.num_elements();
    int64_t chunk_size = (n + pool.size() - 1) / pool.size();

    std::vector<std::future<void>> futures;
    for (int64_t start = 0; start < n; start += chunk_size) {
        int64_t end = std::min(start + chunk_size, n);
        futures.push_back(pool.submit([=]() {
            for (int64_t i = start; i < end; ++i) {
                out_data[i] = a_data[i] + b_data[i];
            }
        }));
    }

    for (auto& f : futures) {
        f.wait();
    }

    return output;
}
```

---

## Summary

Extending PyFlameRT involves:

1. **Custom Operators**: Implement the operator function and register it with `OperatorRegistry`
2. **Custom Backends**: Implement the `Backend` interface for new hardware targets
3. **Custom Model Formats**: Implement `ModelLoader`/`ModelSaver` interfaces
4. **Serving Infrastructure**: Add custom HTTP handlers, metrics, and processing hooks
5. **Optimizations**: Add graph optimization passes and operator fusion

For questions or contributions, see the [Contributing Guide](../CONTRIBUTING.md) or open an issue on GitHub.
