# PyFlameRT Examples and Tutorials

Practical examples and step-by-step tutorials for using PyFlameRT.

---

## Table of Contents

1. [Quick Start Examples](#quick-start-examples)
   - [Basic Inference](#basic-inference)
   - [Batch Processing](#batch-processing)
   - [Multi-Threading Configuration](#multi-threading-configuration)
2. [Working with Models](#working-with-models)
   - [Model Inspection](#model-inspection)
   - [Dynamic Input Shapes](#dynamic-input-shapes)
   - [Multiple Inputs and Outputs](#multiple-inputs-and-outputs)
3. [Performance Optimization](#performance-optimization)
   - [Profiling Inference](#profiling-inference)
   - [Memory Optimization](#memory-optimization)
   - [Benchmarking](#benchmarking)
4. [Model Quantization](#model-quantization)
   - [FP16 Quantization](#fp16-quantization)
   - [INT8 Dynamic Quantization](#int8-dynamic-quantization)
   - [INT8 Static Quantization with Calibration](#int8-static-quantization-with-calibration)
   - [Comparing Quantization Accuracy](#comparing-quantization-accuracy)
5. [Error Handling](#error-handling)
   - [Input Validation](#input-validation)
   - [Graceful Error Recovery](#graceful-error-recovery)
6. [Integration Patterns](#integration-patterns)
   - [REST API Server](#rest-api-server)
   - [Streaming Inference](#streaming-inference)
   - [Pipeline Processing](#pipeline-processing)
7. [Model Serving](#model-serving)
   - [Starting a Model Server](#starting-a-model-server)
   - [Using the Python Client](#using-the-python-client)
   - [Async Inference Client](#async-inference-client)
   - [Kubernetes Deployment](#kubernetes-deployment)
   - [Monitoring with Prometheus](#monitoring-with-prometheus)
8. [C++ Examples](#c-examples)
   - [Basic C++ Usage](#basic-c-usage)
   - [Building Custom Applications](#building-custom-applications)

---

## Quick Start Examples

### Basic Inference

The simplest way to run inference with PyFlameRT:

```python
import numpy as np
import pyflame_rt

# Load the model
session = pyflame_rt.InferenceSession("model.pfm")

# Check model inputs
inputs = session.get_inputs()
print(f"Model expects input '{inputs[0].name}' with shape {inputs[0].shape}")

# Prepare input data
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

# Run inference
outputs = session.run(None, {inputs[0].name: input_data})

# Process results
print(f"Output shape: {outputs[0].shape}")
print(f"Output values: {outputs[0][:5]}")  # First 5 values
```

### Batch Processing

Process multiple samples efficiently:

```python
import numpy as np
import pyflame_rt

def process_batch(session, images):
    """Process a batch of images through the model."""
    # Stack images into a batch
    batch = np.stack(images, axis=0).astype(np.float32)

    # Run inference
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: batch})

    return outputs[0]

# Load model
session = pyflame_rt.InferenceSession("model.pfm")

# Process images in batches
batch_size = 32
images = [np.random.randn(3, 224, 224) for _ in range(100)]

results = []
for i in range(0, len(images), batch_size):
    batch = images[i:i + batch_size]
    batch_results = process_batch(session, batch)
    results.extend(batch_results)

print(f"Processed {len(results)} images")
```

### Multi-Threading Configuration

Configure threading for optimal performance:

```python
import pyflame_rt
import os

# Method 1: Explicit thread count
opts = pyflame_rt.SessionOptions()
opts.num_threads = 4  # Use 4 threads
session = pyflame_rt.InferenceSession("model.pfm", opts)

# Method 2: Use all available cores
opts = pyflame_rt.SessionOptions()
opts.num_threads = os.cpu_count()
session = pyflame_rt.InferenceSession("model.pfm", opts)

# Method 3: Auto (let PyFlameRT decide)
opts = pyflame_rt.SessionOptions()
opts.num_threads = 0  # 0 = automatic
session = pyflame_rt.InferenceSession("model.pfm", opts)

# Method 4: Single-threaded (for debugging)
opts = pyflame_rt.SessionOptions()
opts.num_threads = 1
opts.execution_mode = "sequential"
session = pyflame_rt.InferenceSession("model.pfm", opts)
```

---

## Working with Models

### Model Inspection

Inspect model structure before running inference:

```python
import pyflame_rt

def inspect_model(model_path):
    """Print detailed information about a model."""
    session = pyflame_rt.InferenceSession(model_path)

    # Model metadata
    meta = session.get_modelmeta()
    print("=== Model Metadata ===")
    print(f"  Producer: {meta.producer_name} v{meta.producer_version}")
    print(f"  Domain: {meta.domain}")
    print(f"  Description: {meta.description}")
    if meta.custom_metadata:
        print(f"  Custom metadata: {meta.custom_metadata}")

    # Inputs
    print("\n=== Inputs ===")
    for inp in session.get_inputs():
        shape_str = format_shape(inp.shape)
        print(f"  {inp.name}: {inp.type} {shape_str}")

    # Outputs
    print("\n=== Outputs ===")
    for out in session.get_outputs():
        shape_str = format_shape(out.shape)
        print(f"  {out.name}: {out.type} {shape_str}")

    # Providers
    print("\n=== Execution Providers ===")
    for provider in session.get_providers():
        print(f"  {provider}")

def format_shape(shape):
    """Format shape with dynamic dimensions marked."""
    dims = []
    for d in shape:
        if d is None:
            dims.append("?")
        else:
            dims.append(str(d))
    return "[" + ", ".join(dims) + "]"

# Usage
inspect_model("model.pfm")
```

Output:
```
=== Model Metadata ===
  Producer: PyFlameRT v0.1.0
  Domain: ai.pyflame
  Description: Image classification model

=== Inputs ===
  input: tensor(float) [?, 3, 224, 224]

=== Outputs ===
  output: tensor(float) [?, 1000]

=== Execution Providers ===
  CPUExecutionProvider
```

### Dynamic Input Shapes

Handle models with dynamic dimensions:

```python
import numpy as np
import pyflame_rt

session = pyflame_rt.InferenceSession("model.pfm")

def get_concrete_shape(shape, batch_size=1):
    """Replace dynamic dimensions with concrete values."""
    return [batch_size if d is None else d for d in shape]

# Get input shape with dynamic batch
input_info = session.get_inputs()[0]
print(f"Model input shape: {input_info.shape}")

# Run with different batch sizes
for batch_size in [1, 4, 16, 32]:
    shape = get_concrete_shape(input_info.shape, batch_size)
    input_data = np.random.randn(*shape).astype(np.float32)

    outputs = session.run(None, {input_info.name: input_data})
    print(f"Batch {batch_size}: output shape = {outputs[0].shape}")
```

### Multiple Inputs and Outputs

Work with models that have multiple inputs/outputs:

```python
import numpy as np
import pyflame_rt

# Load model with multiple inputs
session = pyflame_rt.InferenceSession("multi_input_model.pfm")

# Prepare all inputs
inputs = session.get_inputs()
input_feed = {}

for inp in inputs:
    # Create appropriate shape for each input
    shape = [1 if d is None else d for d in inp.shape]

    # Use appropriate dtype
    if "float" in inp.type:
        data = np.random.randn(*shape).astype(np.float32)
    elif "int64" in inp.type:
        data = np.random.randint(0, 100, size=shape).astype(np.int64)
    else:
        data = np.random.randn(*shape).astype(np.float32)

    input_feed[inp.name] = data
    print(f"Input '{inp.name}': shape={shape}, dtype={data.dtype}")

# Run inference
outputs = session.run(None, input_feed)

# Process outputs
output_names = [o.name for o in session.get_outputs()]
for name, output in zip(output_names, outputs):
    print(f"Output '{name}': shape={output.shape}")

# Run with specific outputs only
specific_outputs = session.run(
    [output_names[0]],  # Only first output
    input_feed
)
```

---

## Performance Optimization

### Profiling Inference

Profile model execution to identify bottlenecks:

```python
import numpy as np
import pyflame_rt
import json

# Enable profiling
opts = pyflame_rt.SessionOptions()
opts.enable_profiling = True
opts.log_level = "verbose"

session = pyflame_rt.InferenceSession("model.pfm", opts)

# Run inference
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
for _ in range(10):  # Warm up
    session.run(None, {"input": input_data})

# Get profiling start time
start_ns = session.get_profiling_start_time_ns()
print(f"Profiling started at: {start_ns} ns")

# Run more iterations
for _ in range(100):
    session.run(None, {"input": input_data})

# End profiling and get results
profile_file = session.end_profiling()
print(f"Profile saved to: {profile_file}")

# Parse profile (if JSON format)
try:
    with open(profile_file) as f:
        profile_data = json.load(f)

    # Analyze results
    for event in profile_data.get("events", []):
        print(f"{event['name']}: {event['dur']}us")
except:
    print("Profile analysis not available")
```

### Memory Optimization

Minimize memory usage during inference:

```python
import numpy as np
import pyflame_rt
import gc

class MemoryEfficientInference:
    def __init__(self, model_path):
        opts = pyflame_rt.SessionOptions()
        opts.memory_limit = 1024 * 1024 * 1024  # 1GB limit
        self.session = pyflame_rt.InferenceSession(model_path, opts)
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, data):
        """Run inference with minimal memory footprint."""
        # Ensure contiguous array to avoid copies
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)

        # Run inference
        outputs = self.session.run(None, {self.input_name: data})

        # Return copy to allow internal buffers to be reused
        result = outputs[0].copy()

        return result

    def predict_streaming(self, data_generator, batch_size=32):
        """Process data in streaming fashion to limit memory."""
        for batch in self._batch_generator(data_generator, batch_size):
            yield self.predict(batch)
            gc.collect()  # Help release memory between batches

    def _batch_generator(self, data_gen, batch_size):
        batch = []
        for item in data_gen:
            batch.append(item)
            if len(batch) == batch_size:
                yield np.stack(batch)
                batch = []
        if batch:
            yield np.stack(batch)

# Usage
inference = MemoryEfficientInference("model.pfm")

# Stream through large dataset
def load_images():
    for i in range(1000):
        yield np.random.randn(3, 224, 224).astype(np.float32)

for result in inference.predict_streaming(load_images(), batch_size=16):
    print(f"Batch result shape: {result.shape}")
```

### Benchmarking

Comprehensive benchmarking utilities:

```python
import numpy as np
import pyflame_rt
import time
from statistics import mean, stdev

def benchmark_model(model_path, input_shape, num_warmup=10, num_iterations=100):
    """Benchmark model inference performance."""
    # Setup
    opts = pyflame_rt.SessionOptions()
    opts.num_threads = 4
    session = pyflame_rt.InferenceSession(model_path, opts)
    input_name = session.get_inputs()[0].name

    # Create input data
    input_data = np.random.randn(*input_shape).astype(np.float32)

    # Warmup
    print(f"Warming up ({num_warmup} iterations)...")
    for _ in range(num_warmup):
        session.run(None, {input_name: input_data})

    # Benchmark
    print(f"Benchmarking ({num_iterations} iterations)...")
    latencies = []

    for _ in range(num_iterations):
        start = time.perf_counter_ns()
        session.run(None, {input_name: input_data})
        end = time.perf_counter_ns()
        latencies.append((end - start) / 1e6)  # Convert to ms

    # Results
    results = {
        "model": model_path,
        "input_shape": input_shape,
        "num_iterations": num_iterations,
        "latency_ms": {
            "mean": mean(latencies),
            "std": stdev(latencies),
            "min": min(latencies),
            "max": max(latencies),
            "p50": sorted(latencies)[len(latencies) // 2],
            "p95": sorted(latencies)[int(len(latencies) * 0.95)],
            "p99": sorted(latencies)[int(len(latencies) * 0.99)],
        },
        "throughput_fps": 1000 / mean(latencies),
    }

    return results

def print_benchmark_results(results):
    """Pretty print benchmark results."""
    print("\n" + "=" * 50)
    print(f"Model: {results['model']}")
    print(f"Input shape: {results['input_shape']}")
    print(f"Iterations: {results['num_iterations']}")
    print("-" * 50)
    print(f"Latency (ms):")
    print(f"  Mean: {results['latency_ms']['mean']:.3f}")
    print(f"  Std:  {results['latency_ms']['std']:.3f}")
    print(f"  Min:  {results['latency_ms']['min']:.3f}")
    print(f"  Max:  {results['latency_ms']['max']:.3f}")
    print(f"  P50:  {results['latency_ms']['p50']:.3f}")
    print(f"  P95:  {results['latency_ms']['p95']:.3f}")
    print(f"  P99:  {results['latency_ms']['p99']:.3f}")
    print("-" * 50)
    print(f"Throughput: {results['throughput_fps']:.1f} FPS")
    print("=" * 50)

# Usage
results = benchmark_model("model.pfm", (1, 3, 224, 224))
print_benchmark_results(results)

# Compare batch sizes
for batch_size in [1, 4, 8, 16, 32]:
    results = benchmark_model("model.pfm", (batch_size, 3, 224, 224))
    print(f"Batch {batch_size}: {results['latency_ms']['mean']:.2f}ms, "
          f"{results['throughput_fps'] * batch_size:.1f} img/s")
```

---

## Model Quantization

Quantization reduces model size and can improve inference performance by using lower-precision arithmetic.

### FP16 Quantization

The simplest quantization approach with minimal accuracy impact:

```python
import numpy as np
import pyflame_rt
from pyflame_rt import quantization

def fp16_quantization_example():
    """Convert a model to FP16 precision."""

    # Configure FP16 quantization
    options = pyflame_rt.SessionOptions()
    options.quantization = quantization.QuantConfig.fp16()
    options.verbose_optimization = True  # See quantization logs

    # Load model with FP16
    session = pyflame_rt.InferenceSession("model.pfm", options)

    # Check quantization was applied
    if session.is_quantized():
        report = session.quantization_report()
        print(f"Quantization mode: {report.mode}")
        print(f"Nodes quantized: {report.nodes_quantized}/{report.nodes_total}")
        print(f"Compression ratio: {report.compression_ratio:.2f}x")
        print(f"Original size: {report.original_size_mb:.2f} MB")
        print(f"Quantized size: {report.quantized_size_mb:.2f} MB")

    # Run inference as normal
    input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
    outputs = session.run(None, {"input": input_data})

    return outputs[0]

# Run example
result = fp16_quantization_example()
print(f"Output shape: {result.shape}")
```

### INT8 Dynamic Quantization

Dynamic quantization computes scale/zero-point at runtime:

```python
import numpy as np
import pyflame_rt
from pyflame_rt import quantization

def dynamic_int8_example():
    """Apply dynamic INT8 quantization."""

    # Configure dynamic INT8
    config = quantization.QuantConfig.dynamic_int8()

    # Optional: customize configuration
    config.granularity = quantization.QuantGranularity.PerTensor
    config.symmetric = True
    config.exclude_ops = ["Softmax"]  # Keep Softmax in FP32

    options = pyflame_rt.SessionOptions()
    options.quantization = config

    # Load and run
    session = pyflame_rt.InferenceSession("model.pfm", options)

    # Verify quantization
    report = session.quantization_report()
    print(f"Compression: {report.compression_ratio:.2f}x")
    print(f"Weights quantized: {report.weights_quantized}")
    print(f"Activations quantized: {report.activations_quantized}")

    # Run inference
    input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
    return session.run(None, {"input": input_data})[0]

result = dynamic_int8_example()
```

### INT8 Static Quantization with Calibration

Static quantization uses calibration data for best accuracy:

```python
import numpy as np
import pyflame_rt
from pyflame_rt import quantization

class ImageDataset:
    """Simple dataset for calibration."""
    def __init__(self, size=100):
        self.size = size
        self.index = 0

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= self.size:
            raise StopIteration
        self.index += 1
        # In production, load real calibration images
        return np.random.randn(1, 3, 224, 224).astype(np.float32)

def static_int8_with_calibration():
    """Apply static INT8 quantization with calibration data."""

    # Create calibration data provider
    calibration_dataset = ImageDataset(size=100)
    dataset_iter = iter(calibration_dataset)

    def get_calibration_batch():
        """Returns one batch of calibration data."""
        try:
            image = next(dataset_iter)
            return {"input": image}
        except StopIteration:
            # Reset iterator for multiple calls
            nonlocal dataset_iter
            dataset_iter = iter(calibration_dataset)
            return {"input": next(dataset_iter)}

    # Configure static INT8 with entropy calibration
    config = quantization.QuantConfig.static_int8(calibration_samples=100)
    config.calibration_method = quantization.CalibrationMethod.Entropy
    config.granularity = quantization.QuantGranularity.PerChannel

    options = pyflame_rt.SessionOptions()
    options.quantization = config
    options.calibration_data = get_calibration_batch
    options.calibration_batches = 100

    # Load model (calibration runs automatically)
    print("Running calibration...")
    session = pyflame_rt.InferenceSession("model.pfm", options)

    # Check results
    report = session.quantization_report()
    print(f"Calibration complete!")
    print(f"Compression ratio: {report.compression_ratio:.2f}x")
    print(f"Memory saved: {report.original_size_mb - report.quantized_size_mb:.2f} MB")

    return session

session = static_int8_with_calibration()
```

### Comparing Quantization Accuracy

Compare accuracy across different quantization modes:

```python
import numpy as np
import pyflame_rt
from pyflame_rt import quantization

def compare_quantization_modes(model_path, test_inputs, num_samples=100):
    """Compare accuracy of different quantization modes."""

    results = {}

    # 1. Baseline (FP32)
    print("Running FP32 baseline...")
    fp32_session = pyflame_rt.InferenceSession(model_path)
    fp32_outputs = []
    for inp in test_inputs[:num_samples]:
        out = fp32_session.run(None, {"input": inp})[0]
        fp32_outputs.append(out)
    results['FP32'] = {'outputs': fp32_outputs, 'size_mb': 0}

    # 2. FP16
    print("Running FP16...")
    options = pyflame_rt.SessionOptions()
    options.quantization = quantization.QuantConfig.fp16()
    fp16_session = pyflame_rt.InferenceSession(model_path, options)
    fp16_outputs = []
    for inp in test_inputs[:num_samples]:
        out = fp16_session.run(None, {"input": inp})[0]
        fp16_outputs.append(out)
    report = fp16_session.quantization_report()
    results['FP16'] = {
        'outputs': fp16_outputs,
        'size_mb': report.quantized_size_mb,
        'compression': report.compression_ratio
    }

    # 3. Dynamic INT8
    print("Running Dynamic INT8...")
    options = pyflame_rt.SessionOptions()
    options.quantization = quantization.QuantConfig.dynamic_int8()
    int8_session = pyflame_rt.InferenceSession(model_path, options)
    int8_outputs = []
    for inp in test_inputs[:num_samples]:
        out = int8_session.run(None, {"input": inp})[0]
        int8_outputs.append(out)
    report = int8_session.quantization_report()
    results['INT8_Dynamic'] = {
        'outputs': int8_outputs,
        'size_mb': report.quantized_size_mb,
        'compression': report.compression_ratio
    }

    # Compute accuracy metrics
    print("\n" + "=" * 60)
    print("Quantization Comparison Results")
    print("=" * 60)

    for mode, data in results.items():
        if mode == 'FP32':
            continue

        # Compare with FP32 baseline
        errors = []
        for fp32_out, quant_out in zip(results['FP32']['outputs'], data['outputs']):
            max_err = np.abs(fp32_out - quant_out).max()
            mean_err = np.abs(fp32_out - quant_out).mean()
            errors.append((max_err, mean_err))

        avg_max_err = np.mean([e[0] for e in errors])
        avg_mean_err = np.mean([e[1] for e in errors])

        print(f"\n{mode}:")
        print(f"  Size: {data['size_mb']:.2f} MB")
        print(f"  Compression: {data['compression']:.2f}x")
        print(f"  Avg Max Error: {avg_max_err:.6f}")
        print(f"  Avg Mean Error: {avg_mean_err:.6f}")

        # Check if outputs still give same predictions (for classification)
        if len(results['FP32']['outputs'][0].shape) == 2:
            fp32_preds = [np.argmax(o) for o in results['FP32']['outputs']]
            quant_preds = [np.argmax(o) for o in data['outputs']]
            accuracy = sum(1 for a, b in zip(fp32_preds, quant_preds) if a == b)
            print(f"  Prediction Match: {accuracy}/{num_samples} ({100*accuracy/num_samples:.1f}%)")

    return results

# Usage
test_inputs = [np.random.randn(1, 3, 224, 224).astype(np.float32) for _ in range(100)]
results = compare_quantization_modes("model.pfm", test_inputs)
```

### Manual Quantization Workflow

For advanced control, use the Quantizer API directly:

```python
import numpy as np
import pyflame_rt
from pyflame_rt import quantization

def manual_quantization_workflow():
    """Demonstrate manual quantization with full control."""

    # Step 1: Load model graph
    importer = pyflame_rt.import_module.ONNXImporter()
    result = importer.import_model("model.onnx")
    if not result.success:
        raise RuntimeError(f"Import failed: {result.error}")
    graph = result.graph
    print(f"Loaded graph with {result.stats.node_count} nodes")

    # Step 2: Configure quantization
    config = quantization.QuantConfig()
    config.mode = quantization.QuantMode.StaticInt8
    config.granularity = quantization.QuantGranularity.PerChannel
    config.calibration_method = quantization.CalibrationMethod.Entropy
    config.symmetric = False
    config.exclude_ops = ["Softmax", "LayerNormalization"]

    # Step 3: Create calibrator and collect statistics
    calibrator = quantization.Calibrator(graph, config)

    print("Collecting calibration statistics...")
    for i in range(100):
        # Load your calibration data
        calib_data = {"input": np.random.randn(1, 3, 224, 224).astype(np.float32)}
        calibrator.observe(calib_data)

        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1} batches")

    # Step 4: Compute quantization parameters
    quant_info = calibrator.compute_quant_params()
    print(f"Computed params for {len(quant_info.tensor_params)} tensors")

    # Step 5: Apply quantization
    quantizer = quantization.Quantizer(config)
    quant_result = quantizer.quantize(graph, quant_info)

    if not quant_result.success:
        raise RuntimeError(f"Quantization failed: {quant_result.error_message}")

    print(f"Quantization complete:")
    print(f"  Nodes quantized: {quant_result.stats.nodes_quantized}")
    print(f"  Nodes skipped: {quant_result.stats.nodes_skipped}")
    print(f"  Compression: {quant_result.stats.compression_ratio():.2f}x")

    # Step 6: Create session with quantized graph
    session = pyflame_rt.InferenceSession(quant_result.quantized_graph)

    return session

session = manual_quantization_workflow()
```

### C++ Quantization Example

```cpp
// quantization_example.cpp
#include <pyflame_rt/session.hpp>
#include <pyflame_rt/quantization/quantizer.hpp>
#include <pyflame_rt/quantization/calibrator.hpp>
#include <iostream>
#include <random>

int main(int argc, char* argv[]) {
    using namespace pyflame_rt;
    using namespace pyflame_rt::quantization;

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.pfm>\n";
        return 1;
    }

    try {
        // Method 1: Automatic quantization via SessionOptions
        {
            std::cout << "=== FP16 Quantization ===\n";

            SessionOptions opts;
            opts.quantization = QuantConfig::fp16();

            InferenceSession session(argv[1], opts);

            if (session.is_quantized()) {
                auto report = session.quantization_report();
                std::cout << "Compression: " << report.compression_ratio << "x\n";
                std::cout << "Size: " << report.quantized_size_mb << " MB\n";
            }
        }

        // Method 2: Manual quantization for more control
        {
            std::cout << "\n=== Static INT8 with Calibration ===\n";

            // Load graph
            auto graph = load_model(argv[1]);

            // Configure
            QuantConfig config = QuantConfig::static_int8(100);
            config.calibration_method = CalibrationMethod::Entropy;
            config.granularity = QuantGranularity::PerChannel;

            // Calibrate
            Calibrator calibrator(*graph, config);

            std::mt19937 rng(42);
            std::normal_distribution<float> dist(0.0f, 1.0f);

            for (size_t i = 0; i < 100; ++i) {
                // Create calibration tensor
                Tensor input({1, 3, 224, 224}, DType::Float32);
                float* data = input.data_ptr<float>();
                for (int64_t j = 0; j < input.num_elements(); ++j) {
                    data[j] = dist(rng);
                }

                calibrator.observe({{"input", input}});
            }

            // Compute params
            GraphQuantInfo quant_info = calibrator.compute_quant_params();

            // Quantize
            Quantizer quantizer(config);
            QuantizationResult result = quantizer.quantize(*graph, quant_info);

            if (result.success) {
                std::cout << "Quantized " << result.stats.nodes_quantized << " nodes\n";
                std::cout << "Compression: " << result.stats.compression_ratio() << "x\n";

                // Create session
                InferenceSession session(std::move(result.quantized_graph));
                std::cout << "Session created successfully\n";
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
```

---

## Error Handling

### Input Validation

Validate inputs before running inference:

```python
import numpy as np
import pyflame_rt

def validate_inputs(session, input_feed):
    """Validate inputs match model expectations."""
    errors = []

    expected_inputs = {inp.name: inp for inp in session.get_inputs()}

    # Check for missing inputs
    for name in expected_inputs:
        if name not in input_feed:
            errors.append(f"Missing required input: '{name}'")

    # Check for extra inputs
    for name in input_feed:
        if name not in expected_inputs:
            errors.append(f"Unknown input: '{name}'")

    # Validate each input
    for name, data in input_feed.items():
        if name not in expected_inputs:
            continue

        expected = expected_inputs[name]

        # Check dtype
        expected_np_dtype = get_numpy_dtype(expected.type)
        if data.dtype != expected_np_dtype:
            errors.append(
                f"Input '{name}': expected dtype {expected_np_dtype}, "
                f"got {data.dtype}"
            )

        # Check shape
        if not shapes_compatible(expected.shape, data.shape):
            errors.append(
                f"Input '{name}': shape {list(data.shape)} incompatible "
                f"with expected {expected.shape}"
            )

    return errors

def get_numpy_dtype(type_str):
    """Convert type string to numpy dtype."""
    if "float" in type_str:
        return np.float32
    elif "double" in type_str:
        return np.float64
    elif "int64" in type_str:
        return np.int64
    elif "int32" in type_str:
        return np.int32
    return np.float32

def shapes_compatible(expected, actual):
    """Check if actual shape is compatible with expected (with dynamic dims)."""
    if len(expected) != len(actual):
        return False
    for e, a in zip(expected, actual):
        if e is not None and e != a:
            return False
    return True

# Usage
session = pyflame_rt.InferenceSession("model.pfm")

# Intentionally wrong input
bad_input = {
    "wrong_name": np.zeros((1, 3, 224, 224), dtype=np.float32)
}

errors = validate_inputs(session, bad_input)
if errors:
    for error in errors:
        print(f"Validation error: {error}")
else:
    session.run(None, bad_input)
```

### Graceful Error Recovery

Handle errors gracefully in production:

```python
import numpy as np
import pyflame_rt
import logging
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobustInference:
    def __init__(self, model_path, fallback_value=None):
        self.model_path = model_path
        self.fallback_value = fallback_value
        self.session = None
        self._load_model()

    def _load_model(self):
        """Load model with error handling."""
        try:
            self.session = pyflame_rt.InferenceSession(self.model_path)
            self.input_name = self.session.get_inputs()[0].name
            logger.info(f"Model loaded: {self.model_path}")
        except pyflame_rt.InvalidModelError as e:
            logger.error(f"Invalid model file: {e}")
            raise
        except FileNotFoundError:
            logger.error(f"Model file not found: {self.model_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def predict(self, data):
        """Run inference with comprehensive error handling."""
        try:
            # Validate input
            if not isinstance(data, np.ndarray):
                raise ValueError(f"Expected numpy array, got {type(data)}")

            if data.dtype != np.float32:
                logger.warning(f"Converting input from {data.dtype} to float32")
                data = data.astype(np.float32)

            # Run inference
            outputs = self.session.run(None, {self.input_name: data})
            return outputs[0]

        except pyflame_rt.ValidationError as e:
            logger.error(f"Input validation failed: {e}")
            if self.fallback_value is not None:
                logger.warning("Using fallback value")
                return self.fallback_value
            raise

        except pyflame_rt.PyFlameRTError as e:
            logger.error(f"Inference error: {e}")
            logger.debug(traceback.format_exc())
            if self.fallback_value is not None:
                return self.fallback_value
            raise

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            logger.debug(traceback.format_exc())
            raise

    def predict_batch_safe(self, batch):
        """Process batch, returning results for successful items."""
        results = []
        errors = []

        for i, item in enumerate(batch):
            try:
                result = self.predict(item[np.newaxis, ...])
                results.append((i, result[0]))
            except Exception as e:
                errors.append((i, str(e)))

        return results, errors

# Usage
inference = RobustInference("model.pfm", fallback_value=np.zeros(1000))

# This won't crash even with bad input
try:
    result = inference.predict(np.zeros((1, 3, 224, 224), dtype=np.float32))
    print(f"Success: {result.shape}")
except Exception as e:
    print(f"Failed: {e}")
```

---

## Integration Patterns

### REST API Server

Build a simple inference API server:

```python
"""
Simple REST API server for PyFlameRT inference.
Usage: python server.py
Then: curl -X POST -H "Content-Type: application/json" \
      -d '{"data": [...]}' http://localhost:8000/predict
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import numpy as np
import pyflame_rt

class InferenceServer:
    def __init__(self, model_path):
        self.session = pyflame_rt.InferenceSession(model_path)
        self.input_info = self.session.get_inputs()[0]

    def predict(self, data):
        """Run inference on input data."""
        input_array = np.array(data, dtype=np.float32)
        outputs = self.session.run(None, {self.input_info.name: input_array})
        return outputs[0].tolist()

# Global server instance
server = None

class RequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/predict':
            try:
                # Read request body
                content_length = int(self.headers['Content-Length'])
                body = self.rfile.read(content_length)
                request = json.loads(body)

                # Run inference
                result = server.predict(request['data'])

                # Send response
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({
                    'status': 'success',
                    'result': result
                }).encode())

            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({
                    'status': 'error',
                    'message': str(e)
                }).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'healthy'}).encode())
        elif self.path == '/model':
            info = {
                'inputs': [{
                    'name': inp.name,
                    'shape': inp.shape,
                    'type': inp.type
                } for inp in server.session.get_inputs()],
                'outputs': [{
                    'name': out.name,
                    'shape': out.shape,
                    'type': out.type
                } for out in server.session.get_outputs()]
            }
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(info).encode())
        else:
            self.send_response(404)
            self.end_headers()

def run_server(model_path, host='0.0.0.0', port=8000):
    global server
    server = InferenceServer(model_path)

    httpd = HTTPServer((host, port), RequestHandler)
    print(f"Server running at http://{host}:{port}")
    print("Endpoints:")
    print("  GET  /health  - Health check")
    print("  GET  /model   - Model info")
    print("  POST /predict - Run inference")
    httpd.serve_forever()

if __name__ == '__main__':
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "model.pfm"
    run_server(model_path)
```

### Streaming Inference

Process streaming data in real-time:

```python
import numpy as np
import pyflame_rt
import queue
import threading
import time

class StreamingInference:
    """Real-time streaming inference processor."""

    def __init__(self, model_path, buffer_size=10):
        self.session = pyflame_rt.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

        self.input_queue = queue.Queue(maxsize=buffer_size)
        self.output_queue = queue.Queue(maxsize=buffer_size)

        self.running = False
        self.worker_thread = None

    def start(self):
        """Start the inference worker thread."""
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker)
        self.worker_thread.daemon = True
        self.worker_thread.start()

    def stop(self):
        """Stop the inference worker thread."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=1.0)

    def _worker(self):
        """Worker thread that processes inference requests."""
        while self.running:
            try:
                # Get input from queue with timeout
                item = self.input_queue.get(timeout=0.1)
                frame_id, data = item

                # Run inference
                start = time.perf_counter()
                outputs = self.session.run(None, {self.input_name: data})
                latency = time.perf_counter() - start

                # Put result in output queue
                self.output_queue.put({
                    'frame_id': frame_id,
                    'result': outputs[0],
                    'latency_ms': latency * 1000
                })

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Inference error: {e}")

    def submit(self, frame_id, data):
        """Submit data for inference (non-blocking)."""
        try:
            self.input_queue.put_nowait((frame_id, data))
            return True
        except queue.Full:
            return False

    def get_result(self, timeout=None):
        """Get inference result (blocking)."""
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None

# Usage example: Video stream processing
def process_video_stream():
    inference = StreamingInference("model.pfm", buffer_size=5)
    inference.start()

    try:
        # Simulate video frames
        for frame_id in range(100):
            # Generate fake frame
            frame = np.random.randn(1, 3, 224, 224).astype(np.float32)

            # Submit for processing
            if inference.submit(frame_id, frame):
                print(f"Submitted frame {frame_id}")
            else:
                print(f"Dropped frame {frame_id} (queue full)")

            # Get any available results
            while True:
                result = inference.get_result(timeout=0)
                if result is None:
                    break
                print(f"Frame {result['frame_id']}: "
                      f"latency={result['latency_ms']:.1f}ms")

            time.sleep(0.033)  # ~30 FPS

    finally:
        inference.stop()

if __name__ == '__main__':
    process_video_stream()
```

### Pipeline Processing

Build multi-stage processing pipelines:

```python
import numpy as np
import pyflame_rt
from concurrent.futures import ThreadPoolExecutor

class InferencePipeline:
    """Multi-stage inference pipeline."""

    def __init__(self):
        self.stages = []
        self.executor = ThreadPoolExecutor(max_workers=4)

    def add_stage(self, name, processor):
        """Add a processing stage."""
        self.stages.append((name, processor))

    def process(self, data):
        """Process data through all stages sequentially."""
        result = data
        timings = {}

        for name, processor in self.stages:
            import time
            start = time.perf_counter()
            result = processor(result)
            timings[name] = (time.perf_counter() - start) * 1000

        return result, timings

    def process_batch_parallel(self, items):
        """Process multiple items in parallel."""
        futures = [
            self.executor.submit(self.process, item)
            for item in items
        ]
        return [f.result() for f in futures]

class ModelStage:
    """Inference stage using PyFlameRT model."""

    def __init__(self, model_path):
        self.session = pyflame_rt.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, data):
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)
        outputs = self.session.run(None, {self.input_name: data})
        return outputs[0]

class PreprocessStage:
    """Image preprocessing stage."""

    def __init__(self, target_size=(224, 224), mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
        self.target_size = target_size
        self.mean = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(std, dtype=np.float32).reshape(3, 1, 1)

    def __call__(self, image):
        # Assume image is HWC uint8
        # Resize (simplified - use proper resize in production)
        # Convert to CHW float32
        if image.shape[-1] == 3:  # HWC to CHW
            image = image.transpose(2, 0, 1)
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        return image[np.newaxis, ...]  # Add batch dim

class PostprocessStage:
    """Classification postprocessing stage."""

    def __init__(self, labels=None, top_k=5):
        self.labels = labels or [f"class_{i}" for i in range(1000)]
        self.top_k = top_k

    def __call__(self, logits):
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()

        # Top-k
        top_indices = np.argsort(probs.flatten())[-self.top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append({
                'class_id': int(idx),
                'label': self.labels[idx],
                'probability': float(probs.flatten()[idx])
            })

        return results

# Usage
pipeline = InferencePipeline()
pipeline.add_stage("preprocess", PreprocessStage())
pipeline.add_stage("inference", ModelStage("model.pfm"))
pipeline.add_stage("postprocess", PostprocessStage())

# Process single image
image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
results, timings = pipeline.process(image)

print("Results:")
for r in results:
    print(f"  {r['label']}: {r['probability']:.4f}")

print("\nTimings:")
for stage, ms in timings.items():
    print(f"  {stage}: {ms:.2f}ms")
```

---

## Model Serving

Deploy PyFlameRT models as scalable HTTP services using the built-in serving infrastructure.

### Starting a Model Server

Launch a model server with Python:

```python
import pyflame_rt

# Configure the server
config = pyflame_rt.serving.ServerConfig()
config.http.host = "0.0.0.0"
config.http.port = 8080
config.http.num_workers = 4
config.enable_metrics = True

# Add a model
model_config = pyflame_rt.serving.ModelConfig()
model_config.name = "image_classifier"
model_config.model_path = "models/resnet50.pfm"
model_config.version = "1"
model_config.enable_batching = True
model_config.max_batch_size = 32
model_config.batch_timeout_us = 5000  # 5ms
config.models.append(model_config)

# Create and start server
server = pyflame_rt.serving.ModelServer(config)

def on_ready():
    print(f"Server ready on port {server.http_port()}")
    print("Endpoints:")
    print("  - POST /v1/models/image_classifier/infer")
    print("  - GET  /v1/models")
    print("  - GET  /health/ready")
    print("  - GET  /metrics")

server.on_ready(on_ready)
server.start()

# Block until Ctrl+C
try:
    server.wait()
except KeyboardInterrupt:
    print("\nShutting down...")
    server.stop()
```

Or use the builder pattern in C++:

```cpp
#include "pyflame_rt/serving/model_server.hpp"

int main() {
    using namespace pyflame_rt::serving;

    auto server = ModelServerBuilder()
        .host("0.0.0.0")
        .port(8080)
        .workers(4)
        .enable_metrics()
        .add_model("image_classifier", "models/resnet50.pfm", "1")
        .enable_batching(32, 5000)
        .build();

    server->on_ready([]() {
        std::cout << "Server is ready!" << std::endl;
    });

    server->start();
    server->wait();  // Block until shutdown

    return 0;
}
```

### Using the Python Client

Send inference requests from Python:

```python
from pyflame_rt.serving import ModelClient
import numpy as np
from PIL import Image

# Connect to the server
client = ModelClient("http://localhost:8080", timeout=30.0)

# Wait for server to be ready
print("Waiting for server...")
if not client.wait_for_ready(timeout=60.0):
    raise RuntimeError("Server not ready")

# List available models
print("\nAvailable models:")
for model in client.list_models():
    print(f"  - {model.name} (ready={model.ready})")

# Get model metadata
meta = client.get_model_metadata("image_classifier")
print(f"\nModel: {meta.name}")
print(f"Inputs: {[f'{i.name}: {i.dtype} {i.shape}' for i in meta.inputs]}")
print(f"Outputs: {[f'{o.name}: {o.dtype} {o.shape}' for o in meta.outputs]}")

# Load and preprocess an image
def preprocess_image(image_path):
    img = Image.open(image_path).resize((224, 224))
    arr = np.array(img).astype(np.float32)
    # Normalize (ImageNet mean/std)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    arr = (arr / 255.0 - mean) / std
    # NHWC -> NCHW
    arr = arr.transpose(2, 0, 1)
    return arr[np.newaxis, ...]  # Add batch dimension

# Run inference
input_tensor = preprocess_image("cat.jpg")
response = client.infer(
    model="image_classifier",
    inputs={"input": input_tensor}
)

if response.success:
    output = response.outputs["output"]
    top5_idx = output[0].argsort()[-5:][::-1]
    print(f"\nTop-5 predictions (latency: {response.latency_ms:.2f}ms):")
    for idx in top5_idx:
        print(f"  Class {idx}: {output[0][idx]:.4f}")
else:
    print(f"Inference failed: {response.error_message}")

# Get model statistics
stats = client.get_model_stats("image_classifier")
print(f"\nModel statistics:")
print(f"  Total requests: {stats.total_requests}")
print(f"  Success rate: {stats.successful_requests / max(1, stats.total_requests) * 100:.1f}%")
print(f"  Avg latency: {stats.avg_latency_ms:.2f}ms")
print(f"  P99 latency: {stats.p99_latency_ms:.2f}ms")
```

### Async Inference Client

Use async/await for high-throughput applications:

```python
import asyncio
import numpy as np
from pyflame_rt.serving import AsyncModelClient

async def benchmark_throughput():
    """Benchmark server throughput with concurrent requests."""

    async with AsyncModelClient(
        "http://localhost:8080",
        timeout=30.0,
        max_connections=100
    ) as client:

        # Wait for server
        while not await client.is_ready():
            await asyncio.sleep(0.5)

        # Generate test inputs
        num_requests = 1000
        inputs = [
            {"input": np.random.randn(1, 3, 224, 224).astype(np.float32)}
            for _ in range(num_requests)
        ]

        # Run benchmark
        print(f"Running {num_requests} requests...")
        start_time = asyncio.get_event_loop().time()

        responses = await client.infer_batch(
            model="image_classifier",
            batch_inputs=inputs,
            max_concurrent=50  # Limit concurrent requests
        )

        elapsed = asyncio.get_event_loop().time() - start_time

        # Calculate metrics
        success_count = sum(1 for r in responses if r.success)
        total_latency = sum(r.latency_ms for r in responses)
        avg_latency = total_latency / len(responses)
        throughput = num_requests / elapsed

        print(f"\nResults:")
        print(f"  Requests: {num_requests}")
        print(f"  Success: {success_count} ({success_count/num_requests*100:.1f}%)")
        print(f"  Total time: {elapsed:.2f}s")
        print(f"  Throughput: {throughput:.1f} req/s")
        print(f"  Avg latency: {avg_latency:.2f}ms")

async def stream_inference():
    """Process a stream of images in real-time."""

    async with AsyncModelClient("http://localhost:8080") as client:

        async def process_image(image_data):
            """Process a single image."""
            response = await client.infer(
                model="image_classifier",
                inputs={"input": image_data}
            )
            return response

        # Simulate streaming images
        image_queue = asyncio.Queue()

        async def producer():
            """Produce images at ~30 FPS."""
            for i in range(100):
                image = np.random.randn(1, 3, 224, 224).astype(np.float32)
                await image_queue.put(image)
                await asyncio.sleep(1/30)  # 30 FPS
            await image_queue.put(None)  # Signal end

        async def consumer():
            """Process images as they arrive."""
            while True:
                image = await image_queue.get()
                if image is None:
                    break
                response = await process_image(image)
                print(f"Frame processed: latency={response.latency_ms:.1f}ms")

        # Run producer and consumer concurrently
        await asyncio.gather(producer(), consumer())

# Run benchmarks
asyncio.run(benchmark_throughput())
```

### Kubernetes Deployment

Deploy to Kubernetes with horizontal auto-scaling:

```bash
# Create the namespace
kubectl create namespace pyflame-rt

# Deploy using kustomize
kubectl apply -k deploy/kubernetes/

# Check deployment status
kubectl -n pyflame-rt get pods -w

# View logs
kubectl -n pyflame-rt logs -f deployment/pyflame-rt-server

# Port-forward for local testing
kubectl -n pyflame-rt port-forward svc/pyflame-rt 8080:8080
```

Custom deployment with ConfigMap overrides:

```yaml
# my-deployment/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
  - ../../deploy/kubernetes/

# Override image tag
images:
  - name: pyflame-rt
    newName: my-registry.com/pyflame-rt
    newTag: v1.0.0

# Override config
configMapGenerator:
  - name: pyflame-rt-config
    behavior: merge
    literals:
      - HTTP_WORKERS=8
      - MAX_BATCH_SIZE=64
      - ENABLE_BATCHING=true

# Override replicas
replicas:
  - name: pyflame-rt-server
    count: 4
```

```bash
# Deploy custom configuration
kubectl apply -k my-deployment/

# Scale manually
kubectl -n pyflame-rt scale deployment/pyflame-rt-server --replicas=6

# Check HPA status
kubectl -n pyflame-rt get hpa
```

### Monitoring with Prometheus

Set up monitoring with Prometheus and Grafana:

```python
"""Example Prometheus queries for PyFlameRT metrics."""

# Request rate by model
rate(pyflame_request_total[5m])

# Error rate
sum(rate(pyflame_request_total{status="error"}[5m])) /
sum(rate(pyflame_request_total[5m]))

# P99 latency
histogram_quantile(0.99,
    sum(rate(pyflame_request_latency_seconds_bucket[5m])) by (le, model))

# Active requests
pyflame_requests_active

# Queue depth
pyflame_queue_size
```

Create a Grafana dashboard:

```json
{
  "title": "PyFlameRT Server",
  "panels": [
    {
      "title": "Request Rate",
      "type": "graph",
      "targets": [
        {"expr": "sum(rate(pyflame_request_total[1m])) by (model)"}
      ]
    },
    {
      "title": "Latency Percentiles",
      "type": "graph",
      "targets": [
        {"expr": "histogram_quantile(0.50, sum(rate(pyflame_request_latency_seconds_bucket[5m])) by (le))"},
        {"expr": "histogram_quantile(0.95, sum(rate(pyflame_request_latency_seconds_bucket[5m])) by (le))"},
        {"expr": "histogram_quantile(0.99, sum(rate(pyflame_request_latency_seconds_bucket[5m])) by (le))"}
      ]
    },
    {
      "title": "Error Rate",
      "type": "singlestat",
      "targets": [
        {"expr": "sum(rate(pyflame_request_total{status='error'}[5m])) / sum(rate(pyflame_request_total[5m]))"}
      ]
    }
  ]
}
```

Set up alerting rules:

```yaml
# prometheus-rules.yaml
groups:
  - name: pyflame-rt
    rules:
      - alert: HighErrorRate
        expr: |
          sum(rate(pyflame_request_total{status="error"}[5m])) /
          sum(rate(pyflame_request_total[5m])) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate in PyFlameRT"

      - alert: HighLatency
        expr: |
          histogram_quantile(0.99,
            sum(rate(pyflame_request_latency_seconds_bucket[5m])) by (le)) > 1
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "P99 latency exceeds 1 second"

      - alert: NoModelsLoaded
        expr: pyflame_model_loaded == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "No models loaded on PyFlameRT server"
```

---

## C++ Examples

### Basic C++ Usage

```cpp
// basic_inference.cpp
#include <pyflame_rt/session.hpp>
#include <pyflame_rt/tensor.hpp>
#include <iostream>
#include <random>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.pfm>\n";
        return 1;
    }

    try {
        // Configure session
        pyflame_rt::SessionOptions opts;
        opts.device = "cpu";
        opts.num_threads = 4;

        // Load model
        pyflame_rt::InferenceSession session(argv[1], opts);

        // Get input info
        auto inputs = session.get_inputs();
        std::cout << "Input: " << inputs[0].name << "\n";

        // Create input tensor
        std::vector<int64_t> shape = {1, 3, 224, 224};
        pyflame_rt::Tensor input(shape, pyflame_rt::DType::Float32);

        // Fill with random data
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        float* data = input.data_ptr<float>();
        for (int64_t i = 0; i < input.num_elements(); ++i) {
            data[i] = dist(rng);
        }

        // Run inference
        auto outputs = session.run({}, {{inputs[0].name, input}});

        // Print output
        std::cout << "Output shape: [";
        for (size_t i = 0; i < outputs[0].shape().size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << outputs[0].shape()[i];
        }
        std::cout << "]\n";

        // Print first 5 values
        const float* out_data = outputs[0].data_ptr<float>();
        std::cout << "First 5 values: ";
        for (int i = 0; i < 5 && i < outputs[0].num_elements(); ++i) {
            std::cout << out_data[i] << " ";
        }
        std::cout << "\n";

    } catch (const pyflame_rt::PyFlameRTError& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
```

### Building Custom Applications

CMake setup for a custom application:

```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.18)
project(MyInferenceApp)

set(CMAKE_CXX_STANDARD 17)

# Find PyFlameRT
find_package(PyFlameRT REQUIRED)

# Or if building from source:
# add_subdirectory(path/to/pyflame_rt)

# Create executable
add_executable(my_app
    main.cpp
    preprocessing.cpp
    postprocessing.cpp
)

target_link_libraries(my_app PRIVATE pyflame_rt)

# Install
install(TARGETS my_app RUNTIME DESTINATION bin)
```

Complete application example:

```cpp
// main.cpp
#include <pyflame_rt/session.hpp>
#include <pyflame_rt/tensor.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <algorithm>

class ImageClassifier {
public:
    ImageClassifier(const std::string& model_path,
                    const std::string& labels_path) {
        // Load model
        pyflame_rt::SessionOptions opts;
        opts.num_threads = 4;
        session_ = std::make_unique<pyflame_rt::InferenceSession>(
            model_path, opts);

        input_name_ = session_->get_inputs()[0].name;

        // Load labels
        load_labels(labels_path);
    }

    struct Prediction {
        int class_id;
        std::string label;
        float probability;
    };

    std::vector<Prediction> classify(const float* image_data,
                                     int height, int width,
                                     int top_k = 5) {
        // Preprocess
        auto input = preprocess(image_data, height, width);

        // Inference
        auto outputs = session_->run({}, {{input_name_, input}});

        // Postprocess
        return postprocess(outputs[0], top_k);
    }

    double benchmark(int iterations = 100) {
        // Create dummy input
        pyflame_rt::Tensor input({1, 3, 224, 224},
                                  pyflame_rt::DType::Float32);

        // Warmup
        for (int i = 0; i < 10; ++i) {
            session_->run({}, {{input_name_, input}});
        }

        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            session_->run({}, {{input_name_, input}});
        }
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end - start);
        return duration.count() / 1000.0 / iterations;  // ms per inference
    }

private:
    std::unique_ptr<pyflame_rt::InferenceSession> session_;
    std::string input_name_;
    std::vector<std::string> labels_;

    void load_labels(const std::string& path) {
        std::ifstream file(path);
        std::string line;
        while (std::getline(file, line)) {
            labels_.push_back(line);
        }
    }

    pyflame_rt::Tensor preprocess(const float* data, int h, int w) {
        pyflame_rt::Tensor tensor({1, 3, 224, 224},
                                   pyflame_rt::DType::Float32);

        // Simplified preprocessing - resize and normalize
        // In production, use proper image processing library
        float* out = tensor.data_ptr<float>();

        // ImageNet normalization
        const float mean[] = {0.485f, 0.456f, 0.406f};
        const float std[] = {0.229f, 0.224f, 0.225f};

        for (int c = 0; c < 3; ++c) {
            for (int y = 0; y < 224; ++y) {
                for (int x = 0; x < 224; ++x) {
                    // Simple nearest neighbor resize
                    int src_y = y * h / 224;
                    int src_x = x * w / 224;
                    float val = data[(src_y * w + src_x) * 3 + c];
                    val = (val / 255.0f - mean[c]) / std[c];
                    out[c * 224 * 224 + y * 224 + x] = val;
                }
            }
        }

        return tensor;
    }

    std::vector<Prediction> postprocess(const pyflame_rt::Tensor& output,
                                         int top_k) {
        const float* data = output.data_ptr<float>();
        int64_t num_classes = output.num_elements();

        // Softmax
        std::vector<float> probs(num_classes);
        float max_val = *std::max_element(data, data + num_classes);
        float sum = 0;
        for (int64_t i = 0; i < num_classes; ++i) {
            probs[i] = std::exp(data[i] - max_val);
            sum += probs[i];
        }
        for (auto& p : probs) p /= sum;

        // Get top-k
        std::vector<int> indices(num_classes);
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(indices.begin(), indices.begin() + top_k,
                         indices.end(),
                         [&probs](int a, int b) {
                             return probs[a] > probs[b];
                         });

        std::vector<Prediction> results;
        for (int i = 0; i < top_k; ++i) {
            int idx = indices[i];
            results.push_back({
                idx,
                idx < labels_.size() ? labels_[idx] : "unknown",
                probs[idx]
            });
        }

        return results;
    }
};

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <model.pfm> <labels.txt> [image]\n";
        return 1;
    }

    try {
        ImageClassifier classifier(argv[1], argv[2]);

        // Run benchmark
        double latency = classifier.benchmark(100);
        std::cout << "Average latency: " << latency << " ms\n";
        std::cout << "Throughput: " << 1000.0 / latency << " FPS\n";

        // Classify dummy image
        std::vector<float> dummy_image(224 * 224 * 3, 0.5f);
        auto predictions = classifier.classify(
            dummy_image.data(), 224, 224, 5);

        std::cout << "\nTop-5 predictions:\n";
        for (const auto& pred : predictions) {
            std::cout << "  " << pred.label
                      << ": " << pred.probability << "\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
```

---

## Summary

This guide covered:

1. **Quick Start**: Basic inference, batch processing, threading
2. **Model Operations**: Inspection, dynamic shapes, multiple I/O
3. **Performance**: Profiling, memory optimization, benchmarking
4. **Error Handling**: Input validation, graceful recovery
5. **Integration**: REST API, streaming, pipelines
6. **Model Serving**: HTTP server, Python client, Kubernetes deployment, monitoring
7. **C++ Usage**: Basic examples and application development

For more details, see the [API Reference](API_Reference.md) and [Developer Guide](Developer_Guide.md).
