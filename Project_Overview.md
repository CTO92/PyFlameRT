# PyFlameRT

## Project Overview

PyFlameRT (PyFlame Runtime) is a high-performance inference runtime for deploying trained models on Cerebras Wafer-Scale Engine (WSE) hardware. It serves as the Cerebras equivalent to NVIDIA's TensorRT and ONNX Runtime GPU, providing optimized model execution, quantization, and production deployment capabilities.

## Purpose

Production ML deployments on NVIDIA hardware typically use TensorRT or ONNX Runtime with CUDA execution providers for optimized inference. PyFlameRT replaces these CUDA-dependent runtimes with a Cerebras-optimized inference engine that:

1. Loads models from multiple formats (PyFlame, ONNX, PyTorch checkpoints)
2. Applies Cerebras-specific optimizations (quantization, fusion, tiling)
3. Generates optimized CSL code for inference workloads
4. Provides low-latency, high-throughput inference APIs
5. Supports production deployment patterns (batching, streaming, serving)

## Target Library Replacements

| Original Library | Version | CUDA Dependencies |
|------------------|---------|-------------------|
| onnxruntime-gpu | >= 1.15.0 | CUDAExecutionProvider, cuDNN |
| tensorrt | >= 8.6.0 | CUDA kernels, cuBLAS, cuDNN |

## Core Components

### 1. Model Loading & Import

Support for multiple model formats with automatic conversion to PyFlame IR.

#### Supported Formats

| Format | Source | Import Method |
|--------|--------|---------------|
| **PyFlame** | Native PyFlame models | Direct load |
| **ONNX** | Cross-framework standard | `import_onnx()` |
| **PyTorch** | torch.save() checkpoints | `import_pytorch()` |
| **TorchScript** | torch.jit.script models | `import_torchscript()` |
| **SavedModel** | TensorFlow exports | `import_savedmodel()` (future) |

#### Import API

```python
import pyflame_rt as rt

# Load native PyFlame model
session = rt.InferenceSession("model.pfm")

# Import from ONNX
session = rt.InferenceSession.from_onnx("model.onnx")

# Import from PyTorch checkpoint
session = rt.InferenceSession.from_pytorch(
    "model.pt",
    input_shapes={"input": [1, 3, 224, 224]}
)

# Import from TorchScript
session = rt.InferenceSession.from_torchscript("model.pts")
```

### 2. Model Optimization

Cerebras-specific optimizations for inference performance.

#### Optimization Passes

| Optimization | Description | Benefit |
|--------------|-------------|---------|
| **Operator Fusion** | Combine sequential ops into single kernels | Reduce memory traffic |
| **Constant Folding** | Pre-compute static expressions | Eliminate runtime computation |
| **Dead Code Elimination** | Remove unused graph branches | Reduce code size |
| **Layout Optimization** | Optimize tensor layouts for PE mesh | Improve data locality |
| **Quantization** | Reduce precision (FP32→FP16→INT8) | 2-4x memory/compute savings |
| **Weight Pruning** | Remove near-zero weights | Reduce computation |
| **Knowledge Distillation** | Train smaller models | Faster inference |

#### Quantization Support

```python
import pyflame_rt as rt
from pyflame_rt.quantization import QuantConfig

# Post-training quantization
config = QuantConfig(
    mode="dynamic",           # dynamic, static, or qat
    dtype="int8",             # int8, fp16, or bfloat16
    calibration_data=calib_loader,  # For static quantization
    per_channel=True,         # Per-channel vs per-tensor
)

session = rt.InferenceSession.from_onnx(
    "model.onnx",
    optimization_level=rt.OptLevel.ALL,
    quantization=config
)

# Check quantization report
print(session.quantization_report())
```

#### Optimization Levels

```python
class OptLevel:
    NONE = 0        # No optimization
    BASIC = 1       # Constant folding, dead code elimination
    EXTENDED = 2    # + Operator fusion
    ALL = 3         # + Layout optimization, aggressive fusion
```

### 3. Inference Session

High-performance inference execution with batching and streaming support.

#### Session API

```python
import pyflame_rt as rt
import numpy as np

# Create session with options
options = rt.SessionOptions(
    device="wse",                    # wse, wse2, wse3, or cpu
    num_threads=4,                   # For CPU fallback
    enable_profiling=False,
    memory_limit_mb=4096,
    execution_mode="sequential",     # sequential or parallel
)

session = rt.InferenceSession("model.pfm", options)

# Get input/output metadata
inputs = session.get_inputs()
outputs = session.get_outputs()

for inp in inputs:
    print(f"Input: {inp.name}, shape: {inp.shape}, dtype: {inp.dtype}")

# Run inference
input_data = {"input": np.random.randn(1, 3, 224, 224).astype(np.float32)}
results = session.run(output_names=["output"], input_feed=input_data)

# Batch inference
batch_results = session.run_batch(
    output_names=["output"],
    input_feeds=[input_data] * 32,
    max_batch_size=8
)
```

#### Streaming Inference

For real-time applications (voice biometrics, video):

```python
# Create streaming session
stream = session.create_stream(
    chunk_size=16000,        # Audio samples per chunk
    overlap=1600,            # Overlap between chunks
    stateful=True            # Maintain hidden states
)

# Process audio stream
for chunk in audio_chunks:
    result = stream.process(chunk)
    yield result

# Reset state for new input
stream.reset()
```

### 4. Production Serving

Utilities for deploying models in production environments.

#### HTTP Server

```python
from pyflame_rt.serving import ModelServer

server = ModelServer(
    model_path="model.pfm",
    host="0.0.0.0",
    port=8080,
    workers=4,
    max_batch_size=32,
    batch_timeout_ms=10,
)

server.start()  # Blocking
# Or
server.start_background()  # Non-blocking
```

#### gRPC Server

```python
from pyflame_rt.serving import GRPCServer

server = GRPCServer(
    model_path="model.pfm",
    port=50051,
    max_concurrent_rpcs=100,
)

server.start()
```

#### Client SDK

```python
from pyflame_rt.client import InferenceClient

client = InferenceClient("http://localhost:8080")

# Synchronous inference
result = client.infer({"input": input_array})

# Asynchronous inference
future = client.infer_async({"input": input_array})
result = future.result()

# Batch inference
results = client.infer_batch([{"input": arr} for arr in batch])
```

### 5. Benchmarking & Profiling

Tools for performance analysis and optimization.

```python
from pyflame_rt.benchmark import Benchmark

bench = Benchmark(session)

# Latency benchmark
latency = bench.measure_latency(
    input_feed=sample_input,
    warmup_runs=10,
    benchmark_runs=100
)
print(f"P50: {latency.p50_ms:.2f}ms, P99: {latency.p99_ms:.2f}ms")

# Throughput benchmark
throughput = bench.measure_throughput(
    input_feed=sample_input,
    duration_seconds=30,
    batch_size=32
)
print(f"Throughput: {throughput.samples_per_second:.0f} samples/sec")

# Memory profiling
memory = bench.profile_memory()
print(f"Peak memory: {memory.peak_mb:.0f} MB")

# Operation-level profiling
profile = session.run_with_profiling(input_feed)
for op in profile.operations:
    print(f"{op.name}: {op.duration_us:.0f}μs ({op.percentage:.1f}%)")
```

## Cerebras-Specific Considerations

### Compile-Once, Run-Many

Unlike GPU inference where kernels are JIT-compiled, Cerebras requires:

1. **Graph compilation**: Convert IR to CSL, compile to binary
2. **Binary caching**: Cache compiled binaries for fast startup
3. **Shape specialization**: Separate binaries for different input shapes

```python
# Explicit compilation with caching
session = rt.InferenceSession.from_onnx(
    "model.onnx",
    compile_options=rt.CompileOptions(
        cache_dir="./model_cache",
        input_shapes={"input": [1, 3, 224, 224]},
        dynamic_batch=True,  # Support variable batch sizes
    )
)
```

### PE Mesh Utilization

Optimize inference for different model sizes:

| Model Size | Strategy | PE Utilization |
|------------|----------|----------------|
| Small (<100M params) | Batch parallelism | Replicate model, distribute batches |
| Medium (100M-1B) | Hybrid | Model + batch parallelism |
| Large (>1B params) | Model parallelism | Distribute layers across PEs |

### Memory Management

```python
options = rt.SessionOptions(
    memory_limit_mb=4096,
    memory_pool="default",      # default, arena, or custom
    enable_memory_reuse=True,   # Reuse buffers between ops
)
```

### Multi-Model Serving

Deploy multiple models efficiently:

```python
from pyflame_rt.serving import MultiModelServer

server = MultiModelServer(
    models={
        "face_detect": "models/face_detect.pfm",
        "face_embed": "models/face_embed.pfm",
        "voice_embed": "models/voice_embed.pfm",
    },
    shared_memory_mb=8192,  # Shared PE memory pool
)
```

## Implementation Phases

### Phase 1: Core Runtime
- [ ] Project structure and build system
- [ ] PyFlame model loading and execution
- [ ] Basic inference session API
- [ ] CPU reference backend

### Phase 2: Model Import
- [ ] ONNX model import
- [ ] PyTorch checkpoint import
- [ ] TorchScript import
- [ ] Input shape inference

### Phase 3: Optimization Passes
- [ ] Constant folding
- [ ] Dead code elimination
- [ ] Operator fusion (Conv+BN+ReLU, etc.)
- [ ] Layout optimization for PE mesh

### Phase 4: Quantization
- [ ] FP16 inference
- [ ] BFloat16 inference
- [ ] Dynamic INT8 quantization
- [ ] Static INT8 with calibration
- [ ] Per-channel quantization

### Phase 5: Production Features
- [ ] Binary caching and fast startup
- [ ] Dynamic batching
- [ ] Streaming inference API
- [ ] Memory pool management

### Phase 6: Serving Infrastructure
- [ ] HTTP/REST server (FastAPI)
- [ ] gRPC server
- [ ] Client SDKs (Python, C++)
- [ ] Kubernetes deployment manifests
- [ ] Prometheus metrics

### Phase 7: Advanced Optimization
- [ ] Weight pruning
- [ ] Knowledge distillation utilities
- [ ] Custom op registration
- [ ] Graph partitioning for multi-chip

## API Compatibility Goals

PyFlameRT provides familiar APIs inspired by ONNX Runtime:

```python
# ONNX Runtime code
import onnxruntime as ort

session = ort.InferenceSession(
    "model.onnx",
    providers=['CUDAExecutionProvider']
)
outputs = session.run(None, {"input": input_array})

# PyFlameRT equivalent
import pyflame_rt as rt

session = rt.InferenceSession(
    "model.onnx",
    providers=['CerebrasExecutionProvider']
)
outputs = session.run(None, {"input": input_array})
```

## Key Differences from ONNX Runtime / TensorRT

| Aspect | ONNX Runtime / TensorRT | PyFlameRT |
|--------|-------------------------|-----------|
| Compilation | JIT / build-time | Ahead-of-time with caching |
| Dynamic shapes | Native support | Shape-specialized binaries |
| Backend | CUDA kernels | CSL code generation |
| Memory model | Unified GPU memory | Distributed PE SRAM |
| Batching | Implicit | Explicit batch dimension handling |

## Integration with bioID

PyFlameRT enables production deployment of bioID models:

```python
import pyflame_rt as rt

# Load optimized biometric models
face_session = rt.InferenceSession.from_onnx(
    "bioID_face.onnx",
    optimization_level=rt.OptLevel.ALL,
    quantization=rt.QuantConfig(dtype="fp16")
)

voice_session = rt.InferenceSession.from_onnx(
    "bioID_voice.onnx",
    optimization_level=rt.OptLevel.ALL,
    quantization=rt.QuantConfig(dtype="fp16")
)

# Multi-model server for biometric authentication
server = rt.serving.MultiModelServer(
    models={
        "face_detect": face_detect_session,
        "face_embed": face_embed_session,
        "voice_embed": voice_session,
        "liveness": liveness_session,
    }
)
```

## Technology Stack

PyFlameRT follows the same architecture as PyFlame and PyFlameVision: a **C++ core with Python bindings**.

### Architecture Rationale

| Aspect | Why C++ Core |
|--------|--------------|
| **Performance** | Direct memory management, SIMD vectorization, cache optimization |
| **Ecosystem Consistency** | Matches PyFlame/PyFlameVision architecture |
| **Hardware Integration** | Direct interface with Cerebras SDK and CSL code generation |
| **Production Deployment** | Minimal runtime overhead, predictable latency |
| **Code Reuse** | Share types, utilities, and operators with PyFlame |

### Build System & Tools

| Tool | Purpose |
|------|---------|
| **CMake** | Cross-platform build system (matches PyFlame/PyFlameVision) |
| **pybind11** | Python bindings for C++ classes |
| **C++17** | Modern C++ standard with structured bindings, std::optional, etc. |
| **GoogleTest** | C++ unit testing framework |
| **pytest** | Python binding tests |

### Project Structure

```
PyFlameRT/
├── include/pyflame_rt/       # Public C++ headers
│   ├── core/                 # Graph, Node, Tensor
│   ├── backends/             # Backend interfaces
│   ├── io/                   # Model loading
│   └── pyflame_rt.hpp        # Main include
├── src/                      # C++ implementation
│   ├── core/
│   ├── backends/
│   │   ├── cpu/              # CPU reference backend
│   │   └── cerebras/         # Cerebras WSE backend
│   ├── io/
│   └── bindings/             # pybind11 Python bindings
├── python/pyflame_rt/        # Python package
├── tests/                    # C++ and Python tests
├── cmake/                    # CMake modules
├── third_party/              # External dependencies
├── CMakeLists.txt
└── docs/
```

## Dependencies

### C++ Dependencies
- **PyFlame**: Core tensor operations and IR graph (sister project)
- **protobuf**: Model serialization
- **pybind11**: Python bindings
- **Eigen** (optional): Optimized linear algebra for CPU backend

### Python Dependencies (for bindings)
- **NumPy**: Array interface for Python bindings
- **ONNX** (Phase 2): Model format parsing

### Optional Dependencies
- **FastAPI**: HTTP serving (Python)
- **grpcio**: gRPC serving (Python/C++)

## Success Criteria

1. Load and execute all bioID models on Cerebras WSE
2. Achieve <10ms P99 latency for face embedding inference
3. Support FP16 and INT8 quantization with <1% accuracy loss
4. Provide production-ready serving with >1000 QPS
5. Binary caching enables <100ms cold start
6. Migration from ONNX Runtime requires minimal code changes

## References

- [ONNX Runtime documentation](https://onnxruntime.ai/docs/)
- [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [PyFlame architecture documentation](../PyFlame/docs/)
- [Cerebras CSL programming guide](https://docs.cerebras.net/)
