# PyFlameRT Developer Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Architecture Overview](#architecture-overview)
3. [Getting Started](#getting-started)
4. [Core Concepts](#core-concepts)
5. [Model Import](#model-import)
6. [Graph Optimization](#graph-optimization)
7. [Quantization](#quantization)
8. [Production Features](#production-features)
9. [Serving Infrastructure](#serving-infrastructure)
10. [Advanced Optimization](#advanced-optimization)
11. [Python API](#python-api)
12. [C++ API](#c-api)
13. [Model Format](#model-format)
14. [Extending PyFlameRT](#extending-pyflame-rt)
15. [Performance Considerations](#performance-considerations)
16. [Debugging and Troubleshooting](#debugging-and-troubleshooting)
17. [Best Practices](#best-practices)

---

## Introduction

PyFlameRT is a high-performance inference runtime designed for deploying deep learning models on Cerebras Wafer-Scale Engine (WSE) hardware. It provides an **ONNX Runtime-compatible API** that enables seamless migration from existing inference pipelines.

### Key Features

- **Familiar API**: ONNX Runtime-style `InferenceSession` interface
- **High Performance**: C++ core with Python bindings via pybind11
- **Model Import**: Direct import from ONNX, PyTorch, and TorchScript formats
- **Extensible**: Plugin architecture for custom operators and backends
- **Cross-Platform**: Supports Windows, Linux, and macOS
- **Development-Friendly**: CPU reference backend for testing without hardware

### Design Philosophy

PyFlameRT follows the same architectural patterns as PyFlame and PyFlameVision:

1. **C++ Core**: Performance-critical code in C++17
2. **Python Bindings**: Convenient Python API via pybind11
3. **CMake Build System**: Cross-platform builds matching ecosystem tools
4. **Minimal Dependencies**: Core library has few external requirements

---

## Architecture Overview

### Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Python API                              │
│              (pyflame_rt Python package)                     │
├─────────────────────────────────────────────────────────────┤
│                    pybind11 Bindings                         │
│           (bindings/*.cpp → _pyflame_rt.so)                  │
├─────────────────────────────────────────────────────────────┤
│                      C++ Core Library                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐ │
│  │  Types   │  │  Graph   │  │ Registry │  │   Session    │ │
│  │  Tensor  │  │   Node   │  │   Ops    │  │   Options    │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                  Graph Optimization (Phase 3)                │
│  ┌────────────┐ ┌────────┐ ┌──────┐ ┌────────┐ ┌────────┐  │
│  │ConstFold  │ │  DCE   │ │ CSE  │ │ Fusion │ │ Layout │  │
│  └────────────┘ └────────┘ └──────┘ └────────┘ └────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         PassManager + PatternMatcher                  │  │
│  └──────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                   Quantization (Phase 4)                     │
│  ┌─────────┐ ┌──────────┐ ┌───────────┐ ┌──────────────┐   │
│  │  FP16   │ │ BFloat16 │ │Dynamic INT8│ │ Static INT8 │   │
│  └─────────┘ └──────────┘ └───────────┘ └──────────────┘   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │       Quantizer + Calibrator + QuantParams            │  │
│  └──────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│               Production Features (Phase 5)                  │
│  ┌───────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐  │
│  │MemoryPool │ │  Cache   │ │ Batching │ │  Streaming   │  │
│  │   Arena   │ │  MMap    │ │ Dynamic  │ │    Async     │  │
│  └───────────┘ └──────────┘ └──────────┘ └──────────────┘  │
├─────────────────────────────────────────────────────────────┤
│              Advanced Optimization (Phase 7)                 │
│  ┌───────────┐ ┌────────────┐ ┌──────────┐ ┌────────────┐  │
│  │  Pruning  │ │Distillation│ │ CustomOp │ │ Partition  │  │
│  │ Sparsity  │ │  Teacher   │ │ Registry │ │ Multi-Chip │  │
│  └───────────┘ └────────────┘ └──────────┘ └────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                       Backends                               │
│  ┌──────────────────┐  ┌──────────────────────────────────┐ │
│  │   CPU Backend    │  │   Cerebras Backend (Future)      │ │
│  │  (Reference)     │  │   (WSE/WSE2/WSE3)                │ │
│  └──────────────────┘  └──────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                      Model I/O                               │
│  ┌──────────────────┐  ┌──────────────────────────────────┐ │
│  │   .pfm Format    │  │   Model Import (Phase 2)         │ │
│  └──────────────────┘  └──────────────────────────────────┘ │
│                                │                             │
│            ┌───────────────────┼───────────────────┐        │
│            │                   │                   │        │
│     ┌──────▼─────┐      ┌──────▼─────┐     ┌──────▼─────┐   │
│     │   ONNX     │      │  PyTorch   │     │TorchScript │   │
│     │  Importer  │      │  Importer  │     │  Importer  │   │
│     └────────────┘      └────────────┘     └────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility |
|-----------|----------------|
| **Types** | Data type definitions, shapes, metadata |
| **Tensor** | N-dimensional data container with memory management |
| **Node** | Single operation in computational graph |
| **Graph** | DAG of nodes with inputs, outputs, and initializers |
| **Registry** | Maps operator names to implementations |
| **Backend** | Executes graphs on specific hardware |
| **Session** | High-level API for model loading and inference |
| **Loader** | Deserializes model files into Graph objects |
| **Import** | Converts external model formats (ONNX, PyTorch, TorchScript) to Graph |
| **ShapeInference** | Propagates tensor shapes through the graph |
| **PassManager** | Orchestrates graph optimization passes |
| **PatternMatcher** | Detects graph patterns for fusion and transformation |
| **Optimization Passes** | Transform graphs to improve performance (ConstFold, DCE, CSE, Fusion, Layout) |
| **Quantizer** | Transforms graphs for lower-precision inference (FP16, BF16, INT8) |
| **Calibrator** | Collects statistics from calibration data for static quantization |
| **QuantParams** | Stores scale and zero-point values for quantized tensors |
| **MemoryPool** | Efficient memory allocation with size-class pooling and arenas |
| **BinaryCache** | Caches compiled model artifacts for fast startup |
| **MMapLoader** | Memory-mapped file loading for zero-copy model access |
| **DynamicBatcher** | Automatic request batching for throughput optimization |
| **AsyncSession** | Asynchronous inference with multiple execution streams |

### Data Flow

```
Model File (.pfm/.onnx/.pt)
       │
       ▼
┌──────────────┐
│Loader/Import │ ─── Deserialize model
└──────────────┘
       │
       ▼
┌──────────────┐
│    Graph     │ ─── Validate structure
└──────────────┘
       │
       ▼
┌──────────────┐
│ Optimization │ ─── Run optimization passes
│ (PassManager)│     (ConstFold, DCE, CSE, Fusion)
└──────────────┘
       │
       ▼
┌──────────────┐
│ Quantization │ ─── Apply quantization (if configured)
│ (Quantizer)  │     (FP16, BF16, INT8 dynamic/static)
└──────────────┘
       │
       ▼
┌──────────────┐
│   Backend    │ ─── Select execution provider
└──────────────┘
       │
       ▼
┌──────────────┐
│   Session    │ ─── Ready for inference
└──────────────┘
       │
       ▼
  run(inputs)
       │
       ▼
┌──────────────┐
│   Executor   │ ─── Topological execution
└──────────────┘
       │
       ▼
    outputs
```

---

## Getting Started

### Prerequisites

**Required:**
- CMake 3.18 or higher
- C++17 compatible compiler:
  - GCC 8+ (Linux)
  - Clang 8+ (macOS)
  - MSVC 2019+ (Windows)
- Python 3.9 or higher
- NumPy 1.21 or higher

**Optional:**
- Eigen 3.3+ (for optimized linear algebra)
- Protocol Buffers (for alternative serialization)

### Building from Source

#### Linux/macOS

```bash
# Clone repository
git clone https://github.com/pyflame/pyflame-rt.git
cd pyflame-rt

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYFLAME_RT_BUILD_TESTS=ON \
    -DPYFLAME_RT_BUILD_PYTHON=ON

# Build
cmake --build . --parallel $(nproc)

# Run tests
ctest --output-on-failure

# Install C++ library (optional)
sudo cmake --install .
```

#### Windows

```powershell
# Clone repository
git clone https://github.com/pyflame/pyflame-rt.git
cd pyflame-rt

# Create build directory
mkdir build
cd build

# Configure with CMake (Visual Studio 2019)
cmake .. -G "Visual Studio 16 2019" -A x64 `
    -DPYFLAME_RT_BUILD_TESTS=ON `
    -DPYFLAME_RT_BUILD_PYTHON=ON

# Build
cmake --build . --config Release --parallel

# Run tests
ctest -C Release --output-on-failure
```

### Installing Python Package

```bash
# Development install (editable)
pip install -e .

# Production install
pip install .

# Verify installation
python -c "import pyflame_rt; print(pyflame_rt.__version__)"
```

### Quick Verification

```python
import numpy as np
import pyflame_rt

# Check version
print(f"PyFlameRT version: {pyflame_rt.__version__}")

# Test tensor operations
arr = np.random.randn(2, 3).astype(np.float32)
tensor = pyflame_rt.from_numpy(arr)
print(f"Tensor shape: {tensor.shape}, dtype: {tensor.dtype}")

# Convert back
result = tensor.numpy()
assert np.allclose(arr, result)
print("Installation verified successfully!")
```

---

## Core Concepts

### Data Types (DType)

PyFlameRT supports the following data types:

| DType | Size | NumPy Equivalent | Description |
|-------|------|------------------|-------------|
| `Float32` | 4 bytes | `np.float32` | Standard single precision |
| `Float16` | 2 bytes | `np.float16` | Half precision |
| `BFloat16` | 2 bytes | - | Brain floating point |
| `Float64` | 8 bytes | `np.float64` | Double precision |
| `Int64` | 8 bytes | `np.int64` | 64-bit signed integer |
| `Int32` | 4 bytes | `np.int32` | 32-bit signed integer |
| `Int16` | 2 bytes | `np.int16` | 16-bit signed integer |
| `Int8` | 1 byte | `np.int8` | 8-bit signed integer |
| `UInt8` | 1 byte | `np.uint8` | 8-bit unsigned integer |
| `Bool` | 1 byte | `np.bool_` | Boolean |

```python
import pyflame_rt

# Access dtype enum
print(pyflame_rt.DType.Float32)  # DType.Float32
print(pyflame_rt.DType.Float32.value)  # 0
```

### Tensors

Tensors are the fundamental data container in PyFlameRT.

#### Creating Tensors

```python
import numpy as np
import pyflame_rt

# From NumPy array (recommended)
arr = np.random.randn(1, 3, 224, 224).astype(np.float32)
tensor = pyflame_rt.from_numpy(arr)

# Create empty tensor
tensor = pyflame_rt.Tensor([2, 3, 4], pyflame_rt.DType.Float32)

# From existing tensor
tensor_copy = tensor.clone()  # Deep copy
```

#### Tensor Properties

```python
tensor = pyflame_rt.from_numpy(np.zeros((2, 3, 4), dtype=np.float32))

print(tensor.shape)        # [2, 3, 4]
print(tensor.dtype)        # DType.Float32
print(tensor.ndim)         # 3
print(tensor.num_elements) # 24
print(tensor.size_bytes)   # 96
print(tensor.is_valid)     # True
print(tensor.owns_data)    # True
```

#### Tensor Operations

```python
# Convert to NumPy
arr = tensor.numpy()

# Clone (deep copy)
clone = tensor.clone()

# Reshape
reshaped = tensor.reshape([6, 4])

# Zero out
tensor.zero()
```

### Shapes

Shapes in PyFlameRT support both static and dynamic dimensions.

```python
# Static shape: all dimensions known
static_shape = [1, 3, 224, 224]

# Dynamic shape: use None for unknown dimensions
# (Represented internally with std::optional<int64_t>)
dynamic_shape = [None, 3, 224, 224]  # Dynamic batch
```

### TensorInfo and NodeArg

`TensorInfo` describes tensor metadata without the actual data:

```python
import pyflame_rt

# Create tensor info
info = pyflame_rt.TensorInfo()
info.name = "input"
info.dtype = pyflame_rt.DType.Float32
# info.shape = [1, 3, 224, 224]  # Set via C++

# Check properties
print(info.is_dynamic())  # True if any dim is None
```

`NodeArg` is the ONNX Runtime-compatible descriptor:

```python
# Typically obtained from session
inputs = session.get_inputs()
for inp in inputs:
    print(f"Name: {inp.name}")
    print(f"Shape: {inp.shape}")
    print(f"Type: {inp.type}")  # e.g., "tensor(float32)"
```

### Graph Representation

The computational graph consists of:

1. **Inputs**: Named tensors provided at runtime
2. **Outputs**: Named tensors produced by the graph
3. **Initializers**: Constant tensors (weights, biases)
4. **Nodes**: Operations with inputs, outputs, and attributes

```
Graph: "ResNet"
├── Inputs:
│   └── input: [N, 3, 224, 224] float32
├── Outputs:
│   └── output: [N, 1000] float32
├── Initializers:
│   ├── conv1.weight: [64, 3, 7, 7] float32
│   ├── conv1.bias: [64] float32
│   └── ...
└── Nodes:
    ├── conv1: Conv(input, conv1.weight, conv1.bias) → conv1_out
    ├── relu1: Relu(conv1_out) → relu1_out
    └── ...
```

---

## Model Import

PyFlameRT supports importing models from popular deep learning frameworks directly, eliminating the need for separate conversion tools.

### Supported Formats

| Format | Extension | Source Framework | Features |
|--------|-----------|------------------|----------|
| ONNX | `.onnx` | ONNX ecosystem | Full graph + weights, opsets 9-21 |
| PyTorch | `.pt`, `.pth` | PyTorch | State dict + user-defined graph |
| TorchScript | `.pt` | PyTorch (traced/scripted) | Full graph + weights |

### ONNX Import

ONNX (Open Neural Network Exchange) is the recommended format for importing models from various frameworks.

#### Basic Usage

```python
import pyflame_rt

# Quick import (recommended)
session = pyflame_rt.from_onnx("model.onnx")

# Run inference
import numpy as np
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
results = session.run(None, {"input": input_data})
```

#### Advanced Options

```python
import pyflame_rt

# Create importer with options
options = pyflame_rt.ImportOptions()
options.validate = True           # Validate graph structure
options.infer_shapes = True       # Run shape inference
options.optimize = True           # Apply basic optimizations

importer = pyflame_rt.import_module.ONNXImporter()
result = importer.import_model("model.onnx", options)

if result.success:
    print(f"Imported graph: {result.graph.name}")
    print(f"Nodes: {result.stats.node_count}")
    print(f"Parameters: {result.stats.parameter_count}")

    # Create session from imported graph
    session = pyflame_rt.InferenceSession(result.graph)
else:
    print(f"Import failed: {result.error}")
```

#### Import Statistics

```python
stats = result.stats
print(f"Nodes imported: {stats.node_count}")
print(f"Initializers: {stats.initializer_count}")
print(f"Parameters: {stats.parameter_count}")
print(f"Import time: {stats.import_time_ms:.2f} ms")

# Check for warnings
if stats.warnings:
    for warning in stats.warnings:
        print(f"Warning: {warning}")
```

#### Supported ONNX Operators

PyFlameRT supports 50+ ONNX operators including:

- **Math**: Add, Sub, Mul, Div, MatMul, Gemm, Sqrt, Exp, Log, Pow
- **Activations**: Relu, Sigmoid, Tanh, Softmax, LeakyRelu, Elu, Gelu
- **Tensor Ops**: Reshape, Transpose, Concat, Split, Slice, Squeeze, Unsqueeze
- **Reductions**: ReduceSum, ReduceMean, ReduceMax, ArgMax
- **Neural Network**: Conv, MaxPool, AveragePool, BatchNormalization, Dropout

### PyTorch Import

PyTorch state dictionaries (`.pt`, `.pth`) contain only model weights, not the graph structure. You must provide a model definer function to reconstruct the computation graph.

#### Basic Usage

```python
import pyflame_rt
import numpy as np

def define_simple_mlp(graph, weights):
    """Define a simple MLP model structure."""
    # Add graph inputs
    graph.add_input("input", [None, 784], pyflame_rt.DType.Float32)

    # Add weights as initializers
    graph.add_initializer("fc1.weight", weights["fc1.weight"])
    graph.add_initializer("fc1.bias", weights["fc1.bias"])
    graph.add_initializer("fc2.weight", weights["fc2.weight"])
    graph.add_initializer("fc2.bias", weights["fc2.bias"])

    # Define operations
    graph.add_node("fc1", "Gemm",
                   inputs=["input", "fc1.weight", "fc1.bias"],
                   outputs=["fc1_out"])
    graph.add_node("relu1", "Relu",
                   inputs=["fc1_out"],
                   outputs=["relu1_out"])
    graph.add_node("fc2", "Gemm",
                   inputs=["relu1_out", "fc2.weight", "fc2.bias"],
                   outputs=["output"])

    # Add graph output
    graph.add_output("output", [None, 10], pyflame_rt.DType.Float32)

# Import with model definer
importer = pyflame_rt.import_module.PyTorchImporter()
result = importer.import_model("model.pth", define_simple_mlp)

if result.success:
    session = pyflame_rt.InferenceSession(result.graph)
```

#### Loading Specific Keys

```python
# PyTorch checkpoints often wrap state dicts
importer = pyflame_rt.import_module.PyTorchImporter()

# Try common key patterns
result = importer.import_model("checkpoint.pt", define_model,
                               state_dict_key="model_state_dict")
# Or: state_dict_key="state_dict"
# Or: state_dict_key="model"
```

### TorchScript Import

TorchScript models (created via `torch.jit.trace` or `torch.jit.script`) contain both the graph structure and weights.

#### Basic Usage

```python
import pyflame_rt

importer = pyflame_rt.import_module.TorchScriptImporter()
result = importer.import_model("traced_model.pt")

if result.success:
    session = pyflame_rt.InferenceSession(result.graph)

    import numpy as np
    input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
    results = session.run(None, {"input.1": input_data})
```

#### Handling TorchScript Names

TorchScript often uses numbered input/output names like `input.1`, `output.3`. You can query these from the graph:

```python
result = importer.import_model("model.pt")
graph = result.graph

# Get actual input names
for inp in graph.inputs:
    print(f"Input: {inp.name} - {inp.shape}")

# Get actual output names
for out in graph.outputs:
    print(f"Output: {out.name} - {out.shape}")
```

### Shape Inference

The import module includes automatic shape inference that propagates tensor shapes through the graph.

```python
import pyflame_rt

options = pyflame_rt.ImportOptions()
options.infer_shapes = True  # Enable shape inference

importer = pyflame_rt.import_module.ONNXImporter()
result = importer.import_model("model.onnx", options)

# All tensor shapes are now resolved
for node in result.graph.nodes:
    for output in node.outputs:
        info = result.graph.get_tensor_info(output)
        print(f"{output}: {info.shape}")
```

#### Supported Shape Inference Operators

Shape inference supports 30+ operators:
- Element-wise ops (Add, Mul, etc.) - broadcast semantics
- MatMul/Gemm - matrix multiplication rules
- Conv/Pool - spatial dimension calculation
- Reshape/Transpose - explicit shape changes
- Concat/Split - dimension merging/splitting

### Import Error Handling

```python
import pyflame_rt

try:
    session = pyflame_rt.from_onnx("model.onnx")
except pyflame_rt.InvalidModelError as e:
    print(f"Model file is invalid: {e}")
except pyflame_rt.UnsupportedFormatError as e:
    print(f"Format not supported: {e}")
except pyflame_rt.UnsupportedOperatorError as e:
    print(f"Operator not supported: {e}")
except pyflame_rt.ValidationError as e:
    print(f"Graph validation failed: {e}")
```

### C++ Import API

```cpp
#include <pyflame_rt/import/onnx_importer.hpp>

using namespace pyflame_rt;

// Create importer
ONNXImporter importer;

// Configure options
ImportOptions options;
options.validate = true;
options.infer_shapes = true;

// Import model
ImportResult result = importer.import_model("model.onnx", options);

if (result.success) {
    // Create session from graph
    InferenceSession session(std::move(result.graph));

    // Run inference
    auto outputs = session.run({}, {{"input", input_tensor}});
}
```

---

## Graph Optimization

PyFlameRT includes a comprehensive graph optimization framework that transforms computational graphs to improve inference performance. Optimization runs automatically during session creation based on the configured optimization level.

### Optimization Levels

| Level | Name | Description | Passes Applied |
|-------|------|-------------|----------------|
| 0 | `None` | No optimization | None |
| 1 | `Basic` | Basic optimizations | Constant Folding, DCE |
| 2 | `Extended` | Extended optimizations (default) | + CSE, Operator Fusion |
| 3 | `All` | All optimizations | + Layout Optimization |

### Enabling Optimization

Optimization is enabled by default at the `Extended` level. You can configure it via `SessionOptions`:

```python
import pyflame_rt

# Default: Extended optimization (level 2)
session = pyflame_rt.InferenceSession("model.pfm")

# Disable optimization
options = pyflame_rt.SessionOptions()
options.optimization_level = pyflame_rt.OptLevel.NONE
session = pyflame_rt.InferenceSession("model.pfm", options)

# Enable verbose logging
options = pyflame_rt.SessionOptions()
options.optimization_level = pyflame_rt.OptLevel.ALL
options.verbose_optimization = True
session = pyflame_rt.InferenceSession("model.pfm", options)
```

### Optimization Passes

#### 1. Constant Folding

Pre-computes operations where all inputs are constant (initializers or outputs of other constant operations).

**What it does:**
- Evaluates operations with static inputs at load time
- Replaces computed nodes with constant initializers
- Reduces runtime computation

**Example transformations:**
```
Before: Add(Constant[1,2,3], Constant[4,5,6]) → output
After:  output = Constant[5,7,9]

Before: Shape(input) → Reshape(data, shape_output)
After:  Reshape(data, Constant[1,3,224,224])
```

**Configuration:**
```python
from pyflame_rt import opt

config = opt.ConstantFoldingConfig()
config.max_tensor_bytes = 16 * 1024 * 1024  # Max 16MB tensors
config.fold_shape_ops = True                 # Fold Shape, Gather, etc.
config.fold_expensive_ops = False            # Don't fold MatMul, Conv
config.exclude_ops = ["RandomNormal"]        # Never fold these

# Apply manually
result = opt.fold_constants(graph, config)
print(f"Folded {result.stats.constants_folded} constants")
```

#### 2. Dead Code Elimination (DCE)

Removes nodes that don't contribute to graph outputs.

**What it does:**
- Identifies unused nodes through liveness analysis
- Removes nodes with no consumers
- Cleans up unused initializers
- Removes Identity nodes (pass-through)
- Removes Dropout nodes (no-op in inference)

**Example transformations:**
```
Before: input → Conv → Relu → unused_branch → ...
                    ↘→ output
After:  input → Conv → Relu → output
```

**Configuration:**
```python
config = opt.DCEConfig()
config.remove_initializers = True  # Remove unused weights
config.remove_identity = True      # Remove Identity nodes
config.remove_dropout = True       # Remove Dropout in inference

result = opt.eliminate_dead_code(graph, config)
print(f"Removed {result.stats.nodes_removed} nodes")
print(f"Removed {result.stats.initializers_removed} initializers")
```

#### 3. Common Subexpression Elimination (CSE)

Identifies and shares identical computations.

**What it does:**
- Detects nodes with identical inputs and attributes
- Redirects consumers to a single shared computation
- Reduces redundant operations

**Example transformations:**
```
Before: x = Add(a, b)
        y = Add(a, b)  # Same computation
        z = Mul(x, y)

After:  x = Add(a, b)
        z = Mul(x, x)  # y replaced with x
```

**Configuration:**
```python
config = opt.CSEConfig()
config.check_attributes = True   # Include attributes in comparison
config.max_comparisons = 1000    # Limit comparisons per node

result = opt.eliminate_common_subexpressions(graph, config)
print(f"Eliminated {result.stats.nodes_removed} redundant nodes")
```

#### 4. Operator Fusion

Combines multiple operations into single fused operations for better performance.

**Supported fusion patterns:**

| Pattern | Fused Operation | Description |
|---------|-----------------|-------------|
| Add + Relu | FusedAddRelu | Element-wise add with ReLU activation |
| Mul + Add | FMA | Fused multiply-add |
| MatMul + Add | Gemm | Matrix multiply with bias |
| Conv + Relu | FusedConvRelu | Convolution with ReLU |
| Conv + BatchNorm | FusedConvBN | Fold BN into Conv weights |
| Conv + BN + Relu | FusedConvBNRelu | Three-way fusion |
| BatchNorm + Relu | FusedBNRelu | Batch norm with ReLU |

**Configuration:**
```python
config = opt.FusionConfig()
config.fuse_conv_bn = True               # Conv + BatchNorm
config.fuse_conv_bn_activation = True    # Conv + BN + Relu
config.fuse_matmul_add = True            # MatMul + Add → Gemm
config.fuse_elementwise_activation = True # Add/Mul + Relu
config.check_backend_support = True      # Only fuse if backend supports
config.target_backend = "cpu"            # Target backend name

result = opt.fuse_operators(graph, config)
print(f"Fused {result.stats.nodes_fused} node patterns")
```

#### 5. Layout Optimization

Optimizes tensor memory layouts for target hardware.

**What it does:**
- Analyzes preferred layouts for operations (NCHW vs NHWC)
- Inserts Transpose nodes where layout conversion is needed
- Propagates layout preferences through the graph

**Supported layouts:**

| Layout | Description | Typical Use |
|--------|-------------|-------------|
| `NCHW` | Batch, Channel, Height, Width | PyTorch, most GPUs |
| `NHWC` | Batch, Height, Width, Channel | TensorFlow, some accelerators |
| `NC4HW4` | Blocked format for SIMD | Optimized CPU inference |

**Configuration:**
```python
config = opt.LayoutConfig()
config.conv_layout = opt.Layout.NCHW    # Preferred Conv layout
config.insert_transposes = True          # Insert transpose nodes
config.propagate_layout = True           # Propagate through graph
config.target_backend = "cpu"            # Target backend

# Note: Layout optimization is conservative by default
# and may skip transformation if layouts are already consistent
```

### Using the Pass Manager

For fine-grained control, use the `PassManager` directly:

```python
from pyflame_rt import opt, OptLevel

# Create pass manager with configuration
config = opt.PassManagerConfig()
config.opt_level = OptLevel.EXTENDED
config.max_iterations = 10        # Max fixed-point iterations
config.verbose = True             # Enable logging
config.validate_after_pass = True # Validate after each pass

pm = opt.PassManager(config)

# Register passes (or use create_default)
pm = opt.PassManager.create_default(OptLevel.EXTENDED)

# Run all passes
result = pm.run(graph)

# Run specific pass
result = pm.run_pass("ConstantFoldingPass", graph)

# Run until fixed point (no more changes)
result = pm.run_until_fixed_point(graph)

# Check results
print(f"Modified: {result.modified}")
print(f"Stats: {result.stats}")
for warning in result.warnings:
    print(f"Warning: {warning}")
```

### Optimization Statistics

The `PassResult` object provides detailed statistics:

```python
result = opt.optimize(graph, OptLevel.ALL)

print(f"Nodes removed: {result.stats.nodes_removed}")
print(f"Nodes added: {result.stats.nodes_added}")
print(f"Nodes fused: {result.stats.nodes_fused}")
print(f"Constants folded: {result.stats.constants_folded}")
print(f"Initializers removed: {result.stats.initializers_removed}")
```

### C++ Optimization API

```cpp
#include <pyflame_rt/opt/passes.hpp>

using namespace pyflame_rt;
using namespace pyflame_rt::opt;

// Quick optimization
PassManager pm = PassManager::create_default(OptLevel::Extended);
PassResult result = pm.run_until_fixed_point(graph);

// Custom configuration
PassManagerConfig config;
config.opt_level = OptLevel::All;
config.verbose = true;
config.skip_passes = {"LayoutOptimizationPass"};

PassManager pm(config);
pm.register_pass(std::make_unique<ConstantFoldingPass>());
pm.register_pass(std::make_unique<DeadCodeEliminationPass>());
pm.register_pass(std::make_unique<CSEPass>());
pm.register_pass(std::make_unique<OperatorFusionPass>());

PassResult result = pm.run(graph);

if (result.modified) {
    std::cout << "Removed " << result.stats.nodes_removed << " nodes\n";
}
```

### Custom Optimization Passes

You can implement custom optimization passes:

```cpp
#include <pyflame_rt/opt/pass.hpp>

class MyCustomPass : public Pass {
public:
    const char* name() const override { return "MyCustomPass"; }
    const char* description() const override {
        return "Custom optimization pass";
    }

    PassResult run(Graph& graph) override {
        PassResult result;

        // Iterate through nodes
        for (auto& node : graph.nodes()) {
            // Apply transformations
            if (should_transform(*node)) {
                transform_node(graph, *node);
                result.modified = true;
                result.stats.nodes_removed++;
            }
        }

        return result;
    }

    bool should_run(OptLevel level) const override {
        return level >= OptLevel::Extended;
    }

private:
    bool should_transform(const Node& node) { /* ... */ }
    void transform_node(Graph& graph, Node& node) { /* ... */ }
};

// Register with the pass manager
pm.register_pass(std::make_unique<MyCustomPass>());
```

### Pattern Matching for Fusion

The pattern matcher enables declarative fusion rules:

```cpp
#include <pyflame_rt/opt/pattern_matcher.hpp>

using namespace pyflame_rt::opt;

// Define a pattern: MatMul followed by Add
Pattern pattern;
pattern.add_node(PatternNode::op("MatMul", "matmul"));
pattern.add_node(PatternNode::op("Add", "add"));
pattern.add_edge(PatternEdge::output_to_input("matmul", "add", 0));
pattern.set_root("add");

// Find matches
PatternMatcher matcher;
auto matches = matcher.find_all(graph, pattern);

for (const auto& match : matches) {
    Node* matmul_node = match.get_node("matmul");
    Node* add_node = match.get_node("add");

    // Check single-consumer constraint
    if (match.is_single_consumer("matmul")) {
        // Safe to fuse
        fuse_matmul_add(graph, matmul_node, add_node);
    }
}
```

### Best Practices for Optimization

1. **Start with defaults**: The default `Extended` level works well for most models
2. **Profile first**: Measure inference time before and after optimization
3. **Enable verbose mode**: Use `verbose_optimization = True` to understand what's happening
4. **Check warnings**: Review optimization warnings for potential issues
5. **Validate outputs**: Ensure optimized model produces correct results
6. **Skip problematic passes**: Use `skip_passes` if a pass causes issues

```python
# Recommended optimization workflow
options = pyflame_rt.SessionOptions()
options.optimization_level = pyflame_rt.OptLevel.EXTENDED
options.verbose_optimization = True

session = pyflame_rt.InferenceSession("model.onnx", options)

# Verify outputs match unoptimized version
unoptimized_options = pyflame_rt.SessionOptions()
unoptimized_options.optimization_level = pyflame_rt.OptLevel.NONE
unoptimized_session = pyflame_rt.InferenceSession("model.onnx", unoptimized_options)

test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
opt_output = session.run(None, {"input": test_input})[0]
ref_output = unoptimized_session.run(None, {"input": test_input})[0]

assert np.allclose(opt_output, ref_output, rtol=1e-5, atol=1e-5)
```

---

## Quantization

PyFlameRT provides comprehensive quantization support to reduce model size and improve inference performance. Quantization converts high-precision floating-point models to lower-precision representations while maintaining acceptable accuracy.

### Quantization Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `FP16` | IEEE 754 half-precision (16-bit float) | 2x memory reduction with good accuracy |
| `BFloat16` | Brain floating-point (16-bit) | Training-friendly format with same range as FP32 |
| `DynamicInt8` | Dynamic 8-bit integer quantization | Runtime quantization without calibration |
| `StaticInt8` | Static 8-bit integer with calibration | Best INT8 accuracy with calibration data |

### Quick Start

#### FP16 Quantization

The simplest quantization mode with minimal accuracy loss:

```python
import pyflame_rt
from pyflame_rt import quantization

# Enable FP16 quantization
options = pyflame_rt.SessionOptions()
options.quantization = quantization.QuantConfig.fp16()

session = pyflame_rt.InferenceSession("model.pfm", options)

# Run inference (automatically uses FP16)
results = session.run(None, {"input": input_data})

# Check quantization report
if session.is_quantized():
    report = session.quantization_report()
    print(f"Compression ratio: {report.compression_ratio}x")
```

#### BFloat16 Quantization

BFloat16 preserves the dynamic range of FP32:

```python
options = pyflame_rt.SessionOptions()
options.quantization = quantization.QuantConfig.bfloat16()

session = pyflame_rt.InferenceSession("model.pfm", options)
```

#### Dynamic INT8 Quantization

No calibration required - quantization parameters computed at runtime:

```python
options = pyflame_rt.SessionOptions()
options.quantization = quantization.QuantConfig.dynamic_int8()

session = pyflame_rt.InferenceSession("model.pfm", options)
```

#### Static INT8 Quantization

Best accuracy using calibration data:

```python
import numpy as np

# Define calibration data provider
def get_calibration_data():
    # Return representative sample of your data
    return {
        "input": np.random.randn(1, 3, 224, 224).astype(np.float32)
    }

options = pyflame_rt.SessionOptions()
options.quantization = quantization.QuantConfig.static_int8(calibration_samples=100)
options.calibration_data = get_calibration_data
options.calibration_batches = 100

session = pyflame_rt.InferenceSession("model.pfm", options)
```

### QuantConfig Options

The `QuantConfig` class provides fine-grained control over quantization:

```python
from pyflame_rt.quantization import QuantConfig, QuantMode, QuantGranularity, CalibrationMethod

config = QuantConfig()
config.mode = QuantMode.StaticInt8
config.weight_dtype = pyflame_rt.DType.Int8
config.activation_dtype = pyflame_rt.DType.Int8
config.granularity = QuantGranularity.PerChannel  # or PerTensor
config.calibration_method = CalibrationMethod.Entropy  # MinMax, Entropy, or Percentile
config.symmetric = False  # Asymmetric quantization
config.exclude_ops = ["Softmax", "LayerNormalization"]  # Ops to keep in FP32

options = pyflame_rt.SessionOptions()
options.quantization = config
```

### Quantization Granularity

| Granularity | Description | Trade-off |
|-------------|-------------|-----------|
| `PerTensor` | Single scale/zero-point per tensor | Faster but less accurate |
| `PerChannel` | Scale/zero-point per output channel | More accurate but slightly slower |

```python
# Per-tensor quantization (faster)
config = QuantConfig.dynamic_int8()
config.granularity = QuantGranularity.PerTensor

# Per-channel quantization (more accurate)
config = QuantConfig.static_int8()
config.granularity = QuantGranularity.PerChannel
```

### Calibration Methods

For static INT8 quantization, choose the calibration method based on your model:

| Method | Description | Best For |
|--------|-------------|----------|
| `MinMax` | Uses observed min/max values | Simple models, uniform distributions |
| `Entropy` | KL-divergence minimization | Complex models, CNN feature maps |
| `Percentile` | Uses percentile of values | Models with outliers |

```python
# MinMax calibration (fastest)
config = QuantConfig.static_int8()
config.calibration_method = CalibrationMethod.MinMax

# Entropy calibration (best accuracy for CNNs)
config = QuantConfig.static_int8()
config.calibration_method = CalibrationMethod.Entropy

# Percentile calibration (robust to outliers)
config = QuantConfig.static_int8()
config.calibration_method = CalibrationMethod.Percentile
```

### Manual Quantization API

For advanced use cases, use the `Quantizer` class directly:

```python
from pyflame_rt.quantization import Quantizer, QuantConfig

# Load model graph
importer = pyflame_rt.import_module.ONNXImporter()
result = importer.import_model("model.onnx")
graph = result.graph

# Create quantizer with configuration
config = QuantConfig.dynamic_int8()
quantizer = Quantizer(config)

# Quantize the graph
quant_result = quantizer.quantize_dynamic(graph)

if quant_result.success:
    print(f"Nodes quantized: {quant_result.stats.nodes_quantized}")
    print(f"Compression ratio: {quant_result.stats.compression_ratio():.2f}x")

    # Create session with quantized graph
    session = pyflame_rt.InferenceSession(quant_result.quantized_graph)
```

### Calibration API

For static INT8, use the `Calibrator` class:

```python
from pyflame_rt.quantization import Calibrator, QuantConfig

# Create calibrator
config = QuantConfig.static_int8()
calibrator = Calibrator(graph, config)

# Feed calibration data
for i in range(100):
    calibration_batch = get_calibration_data()
    calibrator.observe(calibration_batch)

# Compute quantization parameters
quant_info = calibrator.compute_quant_params()

# Quantize using calibrated parameters
quantizer = Quantizer(config)
quant_result = quantizer.quantize(graph, quant_info)
```

### Quantization Report

Access detailed quantization statistics:

```python
session = pyflame_rt.InferenceSession("model.pfm", options)

if session.is_quantized():
    report = session.quantization_report()

    print(f"Quantization mode: {report.mode}")
    print(f"Nodes quantized: {report.nodes_quantized} / {report.nodes_total}")
    print(f"Compression ratio: {report.compression_ratio:.2f}x")
    print(f"Original size: {report.original_size_mb:.2f} MB")
    print(f"Quantized size: {report.quantized_size_mb:.2f} MB")
    print(f"Weights quantized: {report.weights_quantized}")
    print(f"Activations quantized: {report.activations_quantized}")
```

### C++ Quantization API

```cpp
#include <pyflame_rt/quantization/quantizer.hpp>
#include <pyflame_rt/quantization/calibrator.hpp>

using namespace pyflame_rt;
using namespace pyflame_rt::quantization;

// FP16 conversion
QuantConfig config = QuantConfig::fp16();
Quantizer quantizer(config);
QuantizationResult result = quantizer.convert_to_fp16(graph);

// Dynamic INT8
config = QuantConfig::dynamic_int8();
quantizer = Quantizer(config);
result = quantizer.quantize_dynamic(graph);

// Static INT8 with calibration
Calibrator calibrator(graph, config);
for (const auto& batch : calibration_data) {
    calibrator.observe(batch);
}
GraphQuantInfo quant_info = calibrator.compute_quant_params();
result = quantizer.quantize(graph, quant_info);

if (result.success) {
    std::cout << "Compression: " << result.stats.compression_ratio() << "x\n";
}
```

### Half-Precision Types

PyFlameRT provides native half-precision types:

```cpp
#include <pyflame_rt/quantization/half_types.hpp>

using namespace pyflame_rt::quantization;

// Float16 (IEEE 754 binary16)
Float16 f16 = Float16::from_float(3.14f);
float f32 = f16.to_float();

// BFloat16 (Brain floating-point)
BFloat16 bf16 = BFloat16::from_float(3.14f);
float f32_2 = bf16.to_float();

// Check special values
if (f16.is_nan()) { /* handle NaN */ }
if (f16.is_inf()) { /* handle infinity */ }
```

### Quantized Operators

PyFlameRT supports quantized versions of common operators:

| Operator | Quantized Version | Notes |
|----------|-------------------|-------|
| `MatMul` | `QuantizedMatMul` | INT8 matrix multiplication |
| `Add` | `QuantizedAdd` | INT8 element-wise addition |
| `Conv` | `QuantizedConv` | INT8 convolution |
| Any | `Quantize` | Convert FP32 to INT8 |
| Any | `Dequantize` | Convert INT8 to FP32 |
| Any | `CastToFP16` | Convert FP32 to FP16 |
| Any | `CastFromFP16` | Convert FP16 to FP32 |

### Best Practices for Quantization

1. **Start with FP16**: Minimal accuracy loss with 2x memory reduction
2. **Use calibration data**: For INT8, use representative data from your use case
3. **Keep sensitive ops in FP32**: Exclude Softmax, LayerNorm from quantization
4. **Validate accuracy**: Always compare quantized outputs with FP32 baseline
5. **Use per-channel for CNNs**: Per-channel quantization works better for convolutions
6. **Profile first**: Measure baseline before quantization to verify benefits

```python
# Recommended workflow
import numpy as np

# 1. Load baseline model
baseline_session = pyflame_rt.InferenceSession("model.pfm")
baseline_output = baseline_session.run(None, {"input": test_input})[0]

# 2. Try FP16 first
options = pyflame_rt.SessionOptions()
options.quantization = quantization.QuantConfig.fp16()
fp16_session = pyflame_rt.InferenceSession("model.pfm", options)
fp16_output = fp16_session.run(None, {"input": test_input})[0]

# 3. Check accuracy
fp16_error = np.abs(baseline_output - fp16_output).max()
print(f"FP16 max error: {fp16_error}")

# 4. If acceptable, measure speedup
# 5. If more compression needed, try INT8 with calibration
```

---

## Production Features

Phase 5 introduces production-ready features for deploying PyFlameRT at scale: memory pool management, binary caching, memory-mapped loading, dynamic batching, and asynchronous inference.

### Memory Pool Management

The memory pool system provides efficient allocation with reduced fragmentation:

#### Pool Configuration

```cpp
#include "pyflame_rt/memory/memory_pool.hpp"
using namespace pyflame_rt::memory;

// Configure the pool
PoolConfig config;
config.size_classes = {64, 128, 256, 512, 1024, 2048, 4096};
config.max_blocks_per_class = 64;
config.alignment = 64;  // Cache line alignment
config.thread_safe = true;
config.track_stats = true;

MemoryPool pool(config);
```

#### Using the Memory Pool

```cpp
// Allocate memory
void* ptr = pool.allocate(256);

// Aligned allocation
void* aligned_ptr = pool.allocate_aligned(1024, 64);

// Deallocate
pool.deallocate(ptr);
pool.deallocate(aligned_ptr);

// Check statistics
auto stats = pool.get_stats();
std::cout << "Pool hit rate: " << stats.hit_rate() << std::endl;
std::cout << "Current usage: " << stats.current_usage << " bytes" << std::endl;
```

#### Arena Allocator

For temporary allocations with bulk reset:

```cpp
Arena arena(1024 * 1024);  // 1MB arena

// Fast linear allocations
void* temp1 = arena.allocate(100);
void* temp2 = arena.allocate(200);
void* temp3 = arena.allocate(300);

std::cout << "Arena used: " << arena.used() << " bytes" << std::endl;

// Reset all at once (very fast)
arena.reset();
```

#### Scoped Arena

RAII-style arena management:

```cpp
Arena arena(1024 * 1024);

{
    ScopedArena scope(arena);
    void* temp = scope.allocate(512);
    // Use temp...
}  // Automatically resets to saved offset

// Arena is back to original state
```

#### Python API

```python
import pyflame_rt

# Configure pool
config = pyflame_rt.memory.PoolConfig()
config.memory_limit = 4 * 1024 * 1024 * 1024  # 4GB
config.alignment = 64

# Create pool
pool = pyflame_rt.memory.MemoryPool(config)

# Check statistics
stats = pool.get_stats()
print(f"Hit rate: {stats.hit_rate():.2%}")
print(f"Peak usage: {stats.peak_usage} bytes")

# Use global pool
pyflame_rt.memory.set_global_pool_config(config)
global_pool = pyflame_rt.memory.global_pool()
```

### Binary Cache System

The binary cache stores compiled model artifacts for fast startup:

#### Cache Configuration

```cpp
#include "pyflame_rt/cache/binary_cache.hpp"
using namespace pyflame_rt::cache;

CacheConfig config;
config.cache_dir = "/tmp/pyflame_cache";
config.max_size_bytes = 1ULL * 1024 * 1024 * 1024;  // 1GB
config.max_entries = 100;
config.ttl = std::chrono::hours(24);  // 24 hour expiration
config.use_mmap = true;

BinaryCache cache(config);
```

#### Using the Cache

```cpp
// Compute cache key
std::string key = CacheKey::compute("model.pfm", session_options);

// Check if cached
if (cache.has(key)) {
    auto graph = cache.get(key);
    // Use cached graph...
} else {
    // Load and cache
    auto graph = load_model("model.pfm");
    cache.put(key, *graph);
}

// Or use get_or_compile for convenience
auto graph = cache.get_or_compile("model.pfm", options, [&]() {
    return compile_model("model.pfm");
});
```

#### Cache Warmup

Pre-populate the cache for known models:

```cpp
std::vector<std::string> model_paths = {
    "model_a.pfm",
    "model_b.pfm",
    "model_c.pfm"
};

cache.warmup(model_paths, session_options);
```

#### Python API

```python
import pyflame_rt
from pathlib import Path

# Configure cache
config = pyflame_rt.cache.CacheConfig()
config.cache_dir = Path("/tmp/pyflame_cache")
config.max_size_bytes = 1024 * 1024 * 1024  # 1GB
config.use_mmap = True

# Create cache
cache = pyflame_rt.cache.BinaryCache(config)

# Check cache status
print(f"Cache entries: {cache.entry_count()}")
print(f"Total size: {cache.total_size()} bytes")

# List entries
for key in cache.list_entries():
    info = cache.get_info(key)
    print(f"  {key}: {info.size_bytes} bytes, valid={info.is_valid}")

# Clear cache
cache.clear()

# Use global cache
pyflame_rt.cache.set_global_cache_config(config)
```

### Memory-Mapped Model Loading

Memory-mapped loading provides fast startup and zero-copy tensor access:

#### Using MMapLoader

```cpp
#include "pyflame_rt/cache/mmap_loader.hpp"
using namespace pyflame_rt::cache;

// Check if file can be memory-mapped
if (MMapLoader::can_mmap("model.pfm")) {
    auto graph = MMapLoader::load("model.pfm");
}

// Open file with memory mapping
auto file = MappedFile::open("weights.bin");
if (file && file->is_valid()) {
    // Provide access hints
    file->advise(MappedFile::Advice::Sequential);

    // Lock in memory for predictable performance
    file->lock();

    // Access data
    const void* data = file->data();
    size_t size = file->size();
}
```

#### Lazy Tensors

Load tensor data on first access:

```cpp
// Create lazy tensor
auto file = std::make_shared<MappedFile>();
*file = std::move(*MappedFile::open("weights.bin"));

LazyTensor lazy(file, offset, shape, DType::Float32);

// Metadata available immediately
std::cout << "Shape: ";
for (auto dim : lazy.shape()) std::cout << dim << " ";
std::cout << std::endl;

// Data loaded on first access
Tensor tensor = lazy.materialize();
```

#### Python API

```python
import pyflame_rt
from pathlib import Path

# Check if memory mapping is supported
path = Path("model.pfm")
if pyflame_rt.cache.MMapLoader.can_mmap(path):
    print("Memory mapping supported")

# Get page alignment
alignment = pyflame_rt.cache.MMapLoader.page_alignment()
print(f"Page alignment: {alignment} bytes")

# Open memory-mapped file
file = pyflame_rt.cache.MappedFile.open(path)
if file.is_valid():
    print(f"File size: {file.size()} bytes")

    # Provide access hints
    file.advise(pyflame_rt.cache.MMapAdvice.Sequential)
```

### Dynamic Batching

Automatic request batching maximizes throughput:

#### Batch Configuration

```cpp
#include "pyflame_rt/batching/dynamic_batcher.hpp"
using namespace pyflame_rt::batching;

BatchConfig config;
config.max_batch_size = 32;
config.max_latency = std::chrono::microseconds(1000);  // 1ms
config.min_batch_size = 1;
config.preferred_sizes = {1, 2, 4, 8, 16, 32};
config.enable_padding = true;
config.num_workers = 4;
config.queue_capacity = 1000;
```

#### Using the Batcher

```cpp
auto session = std::make_shared<InferenceSession>("model.pfm");
DynamicBatcher batcher(session, config);

// Start the batcher
batcher.start();

// Submit requests asynchronously
InferenceRequest req1;
req1.inputs["input"] = tensor1;
auto future1 = batcher.submit(std::move(req1));

InferenceRequest req2;
req2.inputs["input"] = tensor2;
auto future2 = batcher.submit(std::move(req2));

// Get results
auto response1 = future1.get();
auto response2 = future2.get();

// Or use callback
batcher.submit(std::move(req), [](InferenceResponse response) {
    if (response.success) {
        // Process outputs...
    }
});

// Blocking inference
auto response = batcher.infer(std::move(request));

// Check statistics
auto stats = batcher.get_stats();
std::cout << "Avg batch size: " << stats.avg_batch_size << std::endl;
std::cout << "Throughput: " << stats.throughput_rps << " req/s" << std::endl;

// Stop batcher
batcher.stop();
```

#### Priority Batching

Process high-priority requests first:

```cpp
PriorityBatcher batcher(session, config);
batcher.start();

// Submit with priority (higher = sooner)
InferenceRequest urgent;
urgent.inputs["input"] = tensor;
auto future = batcher.submit_priority(std::move(urgent), 10);

InferenceRequest normal;
normal.inputs["input"] = tensor;
auto future2 = batcher.submit_priority(std::move(normal), 1);
```

#### Python API

```python
import numpy as np
import pyflame_rt

# Configure batcher
config = pyflame_rt.batching.BatchConfig()
config.max_batch_size = 32
config.max_latency_us = 1000  # 1ms
config.enable_padding = True

# Create batcher
session = pyflame_rt.InferenceSession("model.pfm")
batcher = pyflame_rt.batching.DynamicBatcher(session, config)

# Start
batcher.start()

# Submit requests
inputs = {"input": np.random.randn(1, 224, 224, 3).astype(np.float32)}
future = batcher.submit(inputs)

# Or blocking
outputs = batcher.infer(inputs)

# With callback
def on_complete(request_id, success, outputs, error):
    if success:
        print(f"Request {request_id} completed")
    else:
        print(f"Request {request_id} failed: {error}")

batcher.submit_with_callback(inputs, on_complete)

# Statistics
stats = batcher.get_stats()
print(f"Avg batch size: {stats.avg_batch_size:.2f}")
print(f"Throughput: {stats.throughput_rps:.2f} req/s")

# Stop
batcher.stop()
```

### Asynchronous Inference

Non-blocking inference with multiple execution streams:

#### Async Options

```cpp
#include "pyflame_rt/streaming/async_session.hpp"
using namespace pyflame_rt::streaming;

AsyncOptions options;
options.num_streams = 4;
options.enable_pipelining = true;
options.callback_threads = 2;
options.max_pending = 100;
options.enable_profiling = true;
```

#### Using AsyncSession

```cpp
auto session = std::make_shared<InferenceSession>("model.pfm");
AsyncSession async_session(session, options);

// Start async processing
async_session.start();

// Submit async request
std::unordered_map<std::string, Tensor> inputs;
inputs["input"] = tensor;
auto future = async_session.run_async({}, inputs);

// Or with callback
async_session.run_async({}, inputs, [](AsyncResult result) {
    if (result.success) {
        std::cout << "Latency: " << result.latency.count() << " us" << std::endl;
    }
});

// Submit to specific stream
auto future2 = async_session.run_on_stream(0, {}, inputs);

// Get result
auto result = future.get();
if (result.success) {
    auto& outputs = result.outputs;
    // Process outputs...
}

// Stream management
std::cout << "Streams: " << async_session.num_streams() << std::endl;
size_t best_stream = async_session.select_stream();

// Synchronize
async_session.synchronize(0);  // Specific stream
async_session.synchronize_all();  // All streams

// Cancel requests
async_session.cancel(request_id);
async_session.cancel_all();

// Statistics
auto stats = async_session.get_stats();
std::cout << "Completed: " << stats.completed_requests << std::endl;
std::cout << "Avg latency: " << stats.avg_latency_us << " us" << std::endl;

// Stop
async_session.stop();
```

#### Streaming Inference

Token-based streaming for sequence models:

```cpp
StreamingInference streaming(session);

// Set callback for each step
streaming.set_step_callback([](const Tensor& output) {
    // Process each token...
});

// Start streaming
std::unordered_map<std::string, Tensor> initial;
initial["input_ids"] = input_tensor;
streaming.start(initial);

// Feed tokens
for (int i = 0; i < max_tokens; ++i) {
    Tensor output = streaming.step(next_token);
    if (is_eos(output)) break;
}

// Get all outputs
auto outputs = streaming.get_outputs();

// Stop
streaming.stop();
```

#### Python API

```python
import numpy as np
import pyflame_rt

# Configure async options
options = pyflame_rt.streaming.AsyncOptions()
options.num_streams = 4
options.enable_profiling = True

# Create async session
session = pyflame_rt.InferenceSession("model.pfm")
async_session = pyflame_rt.streaming.AsyncSession(session, options)

# Start
async_session.start()

# Submit async request
inputs = {"input": np.random.randn(1, 224, 224, 3).astype(np.float32)}
future = async_session.run_async(inputs)

# Or with callback
def on_complete(request_id, success, outputs, error):
    if success:
        print(f"Latency: {outputs.get('latency', 0)} us")

async_session.run_async_callback(inputs, on_complete)

# Get result
result = future.result()  # Python future
if result.success:
    outputs = result.get_outputs()

# Statistics
stats = async_session.get_stats()
print(f"Completed: {stats.completed_requests}")
print(f"Avg latency: {stats.avg_latency_us:.2f} us")

# Stop
async_session.stop()

# Streaming inference
streaming = pyflame_rt.streaming.StreamingInference(session)

def on_token(output):
    print("Got token:", output.shape)

streaming.set_step_callback(on_token)
streaming.start({"input_ids": input_tensor})

for _ in range(max_tokens):
    output = streaming.step(next_token)

outputs = streaming.get_outputs()
streaming.stop()
```

### Best Practices for Production

#### Memory Management

1. **Use memory pools for repeated allocations** of similar sizes
2. **Use arenas for temporary computations** that can be bulk-freed
3. **Enable statistics** during development to tune pool sizes
4. **Set memory limits** to prevent runaway allocations

#### Caching

1. **Enable binary caching** for production deployments
2. **Warmup the cache** on startup with expected models
3. **Set appropriate TTL** based on model update frequency
4. **Monitor cache hit rate** and adjust configuration

#### Batching

1. **Tune max_latency** based on SLA requirements
2. **Set preferred_sizes** to match hardware batch sizes
3. **Use priority batching** for mixed workloads
4. **Monitor queue depth** for capacity planning

#### Async Inference

1. **Match num_streams** to hardware parallelism
2. **Enable profiling** during development
3. **Use callbacks** for non-blocking pipelines
4. **Handle cancellation** gracefully

---

## Serving Infrastructure

Phase 6 introduces production-ready model serving infrastructure including HTTP REST server, model registry with versioning, Prometheus metrics, and deployment tools for Kubernetes.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      Model Server                                │
├─────────────────────────────────────────────────────────────────┤
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐       │
│  │  HTTP Server  │  │ Model Registry│  │    Metrics    │       │
│  │   REST API    │  │  Versioning   │  │  Prometheus   │       │
│  └───────────────┘  └───────────────┘  └───────────────┘       │
├─────────────────────────────────────────────────────────────────┤
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐       │
│  │Model Instance │  │Dynamic Batcher│  │ Health Checks │       │
│  │  (Phase 5)    │  │   (Phase 5)   │  │ Live/Ready    │       │
│  └───────────────┘  └───────────────┘  └───────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

### Model Server

The `ModelServer` class provides a complete serving solution:

#### Basic Usage

```cpp
#include "pyflame_rt/serving/model_server.hpp"
using namespace pyflame_rt::serving;

// Configure server
ServerConfig config;
config.http.host = "0.0.0.0";
config.http.port = 8080;
config.http.num_workers = 4;
config.enable_metrics = true;
config.model_dir = "/models";

// Add models
ModelConfig model;
model.name = "resnet50";
model.model_path = "/models/resnet50.pfm";
model.version = "1";
model.enable_batching = true;
model.max_batch_size = 32;
config.models.push_back(model);

// Create and start server
ModelServer server(config);
server.start();

// Wait for shutdown
server.wait();
```

#### Using the Builder Pattern

```cpp
auto server = ModelServerBuilder()
    .host("0.0.0.0")
    .port(8080)
    .workers(4)
    .enable_metrics()
    .add_model("resnet50", "/models/resnet50.pfm", "1")
    .add_model("bert", "/models/bert.pfm", "1")
    .enable_batching(32, 5000)  // max_batch=32, timeout=5ms
    .build();

server->on_ready([]() {
    std::cout << "Server is ready!" << std::endl;
});

server->start();
```

#### Python Server

```python
import pyflame_rt

# Configure server
config = pyflame_rt.serving.ServerConfig()
config.http.host = "0.0.0.0"
config.http.port = 8080
config.http.num_workers = 4
config.enable_metrics = True

# Add model
model = pyflame_rt.serving.ModelConfig()
model.name = "resnet50"
model.model_path = "/models/resnet50.pfm"
model.version = "1"
model.enable_batching = True
model.max_batch_size = 32
config.models.append(model)

# Create and start server
server = pyflame_rt.serving.ModelServer(config)
server.on_ready(lambda: print("Server ready!"))
server.start()
server.wait()
```

### Model Registry

The `ModelRegistry` manages model loading, versioning, and hot reload:

#### Loading Models

```cpp
ModelRegistry registry;

// Register a model with configuration
ModelConfig config;
config.name = "my_model";
config.model_path = "/path/to/model.pfm";
config.version = "v1.0";
registry.register_model(config);

// Load from path directly
registry.load_from_path("another_model", "/path/to/model.onnx", "v2.0");

// Load all models from directory
registry.load_from_directory("/models");
```

#### Versioning and Access

```cpp
// Get latest version
auto instance = registry.get_latest("my_model");

// Get specific version
auto v1 = registry.get("my_model", "v1.0");
auto v2 = registry.get("my_model", "v2.0");

// List all versions
auto versions = registry.list_versions("my_model");
for (const auto& v : versions) {
    std::cout << v.version << " - " << v.path << std::endl;
}

// Check model existence
if (registry.has("my_model", "v1.0")) {
    // Model is available
}
```

#### Hot Reload

```cpp
// Enable hot reload for file changes
registry.enable_hot_reload(true);

// Set callback for reload events
registry.set_reload_callback([](const std::string& name, const std::string& version) {
    std::cout << "Model reloaded: " << name << "@" << version << std::endl;
});

// Manual reload
registry.reload("my_model", "v1.0");
```

### HTTP REST API

The server exposes a REST API compatible with common inference protocols:

#### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/models/{model}/infer` | Run inference |
| `GET` | `/v1/models` | List all models |
| `GET` | `/v1/models/{model}` | Get model metadata |
| `GET` | `/v1/models/{model}/stats` | Get model statistics |
| `GET` | `/health/live` | Liveness probe |
| `GET` | `/health/ready` | Readiness probe |
| `GET` | `/metrics` | Prometheus metrics |

#### Inference Request Format

```json
POST /v1/models/resnet50/infer
Content-Type: application/json

{
  "request_id": "req-123",
  "inputs": {
    "input": {
      "shape": [1, 3, 224, 224],
      "dtype": "float32",
      "data": "<base64-encoded-data>"
    }
  },
  "outputs": ["output"]
}
```

#### Inference Response Format

```json
{
  "request_id": "req-123",
  "model_name": "resnet50",
  "model_version": "1",
  "outputs": {
    "output": {
      "shape": [1, 1000],
      "dtype": "float32",
      "data": "<base64-encoded-data>"
    }
  },
  "success": true,
  "latency_us": 15234
}
```

### Python Client SDK

The Python client provides both synchronous and asynchronous interfaces:

#### Synchronous Client

```python
from pyflame_rt.serving import ModelClient
import numpy as np

# Create client
client = ModelClient("http://localhost:8080", timeout=30.0)

# Wait for server to be ready
if client.wait_for_ready(timeout=60.0):
    print("Server is ready!")

# List models
models = client.list_models()
for model in models:
    print(f"{model.name}: ready={model.ready}, versions={model.versions}")

# Get model metadata
meta = client.get_model_metadata("resnet50")
print(f"Inputs: {[i.name for i in meta.inputs]}")
print(f"Outputs: {[o.name for o in meta.outputs]}")

# Run inference
input_tensor = np.random.randn(1, 3, 224, 224).astype(np.float32)
response = client.infer(
    model="resnet50",
    inputs={"input": input_tensor}
)

if response.success:
    output = response.outputs["output"]
    print(f"Prediction shape: {output.shape}")
    print(f"Latency: {response.latency_ms:.2f}ms")
else:
    print(f"Error: {response.error_message}")
```

#### Batch Inference

```python
# Process multiple inputs in parallel
batch_inputs = [
    {"input": np.random.randn(1, 3, 224, 224).astype(np.float32)}
    for _ in range(10)
]

responses = client.infer_batch("resnet50", batch_inputs, max_workers=4)
for i, response in enumerate(responses):
    print(f"Request {i}: latency={response.latency_ms:.2f}ms")
```

#### Asynchronous Client

```python
import asyncio
from pyflame_rt.serving import AsyncModelClient
import numpy as np

async def main():
    async with AsyncModelClient("http://localhost:8080") as client:
        # Check health
        if await client.is_ready():
            # Run inference
            input_tensor = np.random.randn(1, 3, 224, 224).astype(np.float32)
            response = await client.infer(
                model="resnet50",
                inputs={"input": input_tensor}
            )
            print(f"Output shape: {response.outputs['output'].shape}")

        # Batch async inference
        inputs = [
            {"input": np.random.randn(1, 3, 224, 224).astype(np.float32)}
            for _ in range(100)
        ]
        responses = await client.infer_batch("resnet50", inputs, max_concurrent=10)

        total_latency = sum(r.latency_ms for r in responses)
        print(f"Average latency: {total_latency / len(responses):.2f}ms")

asyncio.run(main())
```

### Prometheus Metrics

The server exports Prometheus-compatible metrics:

#### Available Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `pyflame_request_total` | Counter | Total requests by model and status |
| `pyflame_request_latency_seconds` | Histogram | Request latency distribution |
| `pyflame_requests_active` | Gauge | Currently active requests |
| `pyflame_model_loaded` | Gauge | Model loaded status (1=loaded, 0=unloaded) |
| `pyflame_batch_size` | Histogram | Batch size distribution |
| `pyflame_queue_size` | Gauge | Current batch queue size |
| `pyflame_inference_errors_total` | Counter | Inference errors by type |

#### Accessing Metrics

```bash
# Fetch metrics
curl http://localhost:8080/metrics

# Example output
# HELP pyflame_request_total Total number of inference requests
# TYPE pyflame_request_total counter
pyflame_request_total{model="resnet50",status="success"} 1523
pyflame_request_total{model="resnet50",status="error"} 12

# HELP pyflame_request_latency_seconds Request latency in seconds
# TYPE pyflame_request_latency_seconds histogram
pyflame_request_latency_seconds_bucket{model="resnet50",le="0.01"} 1200
pyflame_request_latency_seconds_bucket{model="resnet50",le="0.05"} 1500
pyflame_request_latency_seconds_bucket{model="resnet50",le="0.1"} 1520
```

#### Python Metrics Access

```python
from pyflame_rt.serving.metrics import MetricsRegistry

# Get the singleton registry
registry = MetricsRegistry.instance()

# Export as Prometheus format
prometheus_text = registry.export_prometheus()
print(prometheus_text)

# Reset all metrics
registry.reset()
```

### Kubernetes Deployment

PyFlameRT provides Kubernetes manifests for production deployment:

#### Quick Start

```bash
# Deploy to Kubernetes
kubectl apply -k deploy/kubernetes/

# Check deployment status
kubectl -n pyflame-rt get pods
kubectl -n pyflame-rt get services
```

#### Configuration via ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: pyflame-rt-config
  namespace: pyflame-rt
data:
  HTTP_PORT: "8080"
  HTTP_WORKERS: "4"
  ENABLE_BATCHING: "true"
  MAX_BATCH_SIZE: "32"
  MODEL_DIR: "/models"
```

#### Horizontal Pod Autoscaler

The HPA configuration automatically scales based on CPU/memory:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
spec:
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
```

#### Health Probes

The deployment includes liveness and readiness probes:

```yaml
livenessProbe:
  httpGet:
    path: /health/live
    port: 8080
  initialDelaySeconds: 10
  periodSeconds: 15

readinessProbe:
  httpGet:
    path: /health/ready
    port: 8080
  initialDelaySeconds: 5
  periodSeconds: 10
```

### Docker Deployment

Build and run the server using Docker:

```bash
# Build image
docker build -t pyflame-rt:latest -f deploy/docker/Dockerfile .

# Run container
docker run -d \
  -p 8080:8080 \
  -p 9091:9091 \
  -v /path/to/models:/models \
  -e HTTP_PORT=8080 \
  -e ENABLE_BATCHING=true \
  -e MAX_BATCH_SIZE=32 \
  pyflame-rt:latest

# Check health
curl http://localhost:8080/health/ready
```

### Best Practices for Serving

#### Performance

1. **Enable batching** for throughput-sensitive workloads
2. **Tune batch timeout** based on latency requirements
3. **Use multiple workers** matching available CPU cores
4. **Enable warmup requests** to avoid cold start latency

#### Reliability

1. **Configure health checks** for Kubernetes deployments
2. **Set request timeouts** to prevent hung requests
3. **Monitor metrics** for capacity planning
4. **Use model versioning** for safe rollouts

#### Security

1. **Enable TLS** for production deployments
2. **Configure rate limiting** to prevent abuse
3. **Use authentication** for sensitive models
4. **Validate input shapes** to prevent crashes

---

## Advanced Optimization

PyFlameRT provides advanced optimization techniques for model compression and multi-device deployment. Phase 7 includes weight pruning, knowledge distillation, custom operator registration, and graph partitioning for multi-chip execution.

### Weight Pruning

Weight pruning removes less important weights from neural networks to reduce model size and improve inference speed while maintaining accuracy.

#### Pruning Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `Unstructured` | Prune individual weights | Maximum sparsity, requires sparse compute support |
| `Structured` | Prune entire channels/filters | Hardware-friendly, works on any backend |
| `Block` | Prune blocks of weights | Balance between sparsity and hardware efficiency |
| `NM` | N:M sparsity pattern (e.g., 2:4) | Optimized for NVIDIA Ampere+ and WSE |

#### Quick Start

```python
import pyflame_rt
from pyflame_rt import pruning

# Magnitude-based pruning (50% sparsity)
config = pruning.PruningConfig.magnitude_pruning(0.5)
pruner = pruning.WeightPruner(config)

# Load and prune model
session = pyflame_rt.InferenceSession("model.pfm")
graph = session.graph()
pruned_graph = pruner.prune(graph)

# Create new session with pruned model
pruned_session = pyflame_rt.InferenceSession(pruned_graph)
stats = pruner.get_stats()
print(f"Achieved sparsity: {stats.actual_sparsity():.1%}")
print(f"Compression ratio: {stats.compression_ratio():.2f}x")
```

#### Structured Pruning

Remove entire channels or filters for hardware-efficient inference:

```python
# Structured (channel) pruning
config = pruning.PruningConfig.structured_pruning(0.3)  # Remove 30% of channels
pruner = pruning.WeightPruner(config)
pruned_graph = pruner.prune(graph)
```

#### N:M Sparsity

N:M sparsity keeps N values out of every M consecutive values, optimized for modern hardware:

```python
# 2:4 sparsity (50% sparse, hardware-accelerated)
config = pruning.PruningConfig.nm_sparsity(n=2, m=4)
pruner = pruning.WeightPruner(config)
pruned_graph = pruner.prune(graph)
```

#### Gradual Pruning

Apply pruning gradually during fine-tuning for better accuracy:

```python
config = pruning.PruningConfig()
config.target_sparsity = 0.9
config.schedule = pruning.PruningSchedule.CUBIC
config.start_step = 0
config.end_step = 10000
config.initial_sparsity = 0.0

pruner = pruning.WeightPruner(config)

# Get sparsity target at each training step
for step in range(10000):
    current_sparsity = pruner.get_sparsity_at_step(step)
    # Apply pruning masks during training...
```

#### Pruning Configuration Options

```python
config = pruning.PruningConfig()
config.target_sparsity = 0.7          # 70% of weights pruned
config.granularity = pruning.PruningGranularity.UNSTRUCTURED
config.criterion = pruning.PruningCriterion.MAGNITUDE  # or MOVEMENT, TAYLOR, LAMP
config.schedule = pruning.PruningSchedule.ONE_SHOT     # or ITERATIVE, CUBIC
config.block_size = [4, 4]            # For block pruning

# Per-layer sparsity targets
config.per_layer_sparsity = {
    "layer1.weight": 0.5,
    "layer2.weight": 0.8,
}

# Exclude sensitive layers
config.add_exclude_layer("final_classifier")
```

#### Sparse Tensor Operations

Work with sparse tensors for memory-efficient storage:

```python
from pyflame_rt.pruning import SparseTensor, SparseFormat

# Convert dense tensor to sparse
dense_tensor = session.get_initializer("conv1.weight")
sparse = SparseTensor.from_dense(dense_tensor, SparseFormat.CSR)

print(f"Sparsity: {sparse.sparsity():.1%}")
print(f"Memory: {sparse.memory_bytes()} bytes")
print(f"Compression: {sparse.compression_ratio():.2f}x")

# Sparse matrix multiplication
result = pruning.sparse_matmul(sparse, input_tensor)
```

### Knowledge Distillation

Knowledge distillation trains a smaller "student" model to mimic a larger "teacher" model, achieving compression while maintaining accuracy.

#### Quick Start

```python
from pyflame_rt import distillation

# Load teacher model
teacher_session = pyflame_rt.InferenceSession("teacher_model.pfm")
teacher_graph = teacher_session.graph()

# Create half-size student
student_config = distillation.StudentConfig.half_size()
student_graph = distillation.create_student_model(teacher_graph, student_config)

# Configure distillation
config = distillation.DistillationConfig.soft_label(temperature=4.0)
config.alpha = 0.7  # Weight for distillation loss

# Create trainer
trainer = distillation.DistillationTrainer(teacher_graph, student_graph, config)

# Prepare dataset
dataset = distillation.InMemoryDataset()
for batch in training_data:
    dataset.add_sample({"input": batch})

trainer.set_dataset(dataset)

# Train
training_config = distillation.TrainingConfig()
training_config.learning_rate = 1e-4
training_config.num_epochs = 10
training_config.batch_size = 32
trainer.set_training_config(training_config)

result = trainer.train()
print(f"Final loss: {result.final_loss:.4f}")
print(f"Compression: {result.compression_ratio():.2f}x")

# Save trained student
student_session = pyflame_rt.InferenceSession(result.student_graph)
```

#### Distillation Loss Types

| Loss Type | Description | Use Case |
|-----------|-------------|----------|
| `KL_DIVERGENCE` | KL divergence between soft targets | Classification models |
| `MSE` | Mean squared error for features | Feature matching |
| `COSINE` | Cosine similarity loss | Embedding models |
| `ATTENTION` | Attention transfer | Transformer models |
| `HINT` | Hint-based feature distillation | Deep networks |

#### Feature Distillation

Match intermediate layer representations:

```python
config = distillation.DistillationConfig.feature_distillation(
    layers=["encoder.layer.2", "encoder.layer.4", "encoder.layer.6"]
)
config.feature_weights = [0.3, 0.3, 0.4]  # Weights per layer
config.normalize_features = True
```

#### Attention Transfer

Transfer attention patterns for transformer models:

```python
config = distillation.DistillationConfig.attention_transfer(
    layers=["attention.layer.0", "attention.layer.6", "attention.layer.11"]
)
```

#### Student Architecture

Configure student model size:

```python
# Half-size student
student_config = distillation.StudentConfig.half_size()

# Quarter-size student
student_config = distillation.StudentConfig.quarter_size()

# Custom configuration
student_config = distillation.StudentConfig()
student_config.hidden_dim_ratio = 0.5   # Half the hidden dimensions
student_config.num_layers_ratio = 0.5   # Half the layers
student_config.num_heads_ratio = 0.5    # Half the attention heads
```

#### Training Configuration

```python
training_config = distillation.TrainingConfig()
training_config.learning_rate = 1e-4
training_config.batch_size = 32
training_config.num_epochs = 10
training_config.warmup_steps = 1000
training_config.weight_decay = 0.01
training_config.lr_schedule = "cosine"  # or "linear", "constant"
training_config.gradient_clip = 1.0
training_config.early_stopping_patience = 3
training_config.checkpoint_frequency = 500
```

### Custom Operators

Register custom operators for specialized computations not covered by standard operators.

#### Quick Registration

```python
from pyflame_rt import custom
import numpy as np

# Define kernel function
def my_gelu(inputs, attrs):
    x = pyflame_rt.to_numpy(inputs[0])
    # GELU activation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    result = x * 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
    return [pyflame_rt.from_numpy(result)]

# Register the operator
custom.register_custom_op(
    name="MyGELU",
    domain="custom",
    input_names=["X"],
    output_names=["Y"],
    kernel_fn=my_gelu
)
```

#### Builder API

Use the fluent builder API for more control:

```python
from pyflame_rt.custom import CustomOpBuilder, BackendType

def fused_bias_relu(inputs, attrs):
    x = pyflame_rt.to_numpy(inputs[0])
    bias = pyflame_rt.to_numpy(inputs[1])
    result = np.maximum(0, x + bias)  # Add bias and ReLU
    return [pyflame_rt.from_numpy(result)]

def shape_inference(input_shapes):
    return [input_shapes[0]]  # Output same shape as input

op = CustomOpBuilder("FusedBiasRelu") \
    .domain("custom") \
    .version(1) \
    .doc("Fused bias addition and ReLU activation") \
    .input("X", pyflame_rt.DType.Float32) \
    .input("bias", pyflame_rt.DType.Float32) \
    .output("Y", pyflame_rt.DType.Float32) \
    .attr_float("alpha", required=False) \
    .kernel(fused_bias_relu, BackendType.CPU) \
    .shape_inference(shape_inference) \
    .build()
```

#### Operator with Attributes

```python
def scaled_add(inputs, attrs):
    x = pyflame_rt.to_numpy(inputs[0])
    y = pyflame_rt.to_numpy(inputs[1])
    scale = attrs.get("scale", 1.0)
    return [pyflame_rt.from_numpy(x + scale * y)]

CustomOpBuilder("ScaledAdd") \
    .input("X") \
    .input("Y") \
    .output("Z") \
    .attr_float("scale", required=True) \
    .kernel(scaled_add) \
    .build()
```

#### Operator with Gradient

For training support, register a gradient function:

```python
def my_square(inputs, attrs):
    x = pyflame_rt.to_numpy(inputs[0])
    return [pyflame_rt.from_numpy(x ** 2)]

def my_square_grad(inputs, grad_outputs):
    x = pyflame_rt.to_numpy(inputs[0])
    grad_out = pyflame_rt.to_numpy(grad_outputs[0])
    grad_x = 2 * x * grad_out  # d(x^2)/dx = 2x
    return [pyflame_rt.from_numpy(grad_x)]

CustomOpBuilder("MySquare") \
    .input("X") \
    .output("Y") \
    .kernel(my_square) \
    .gradient(my_square_grad) \
    .build()
```

#### Registry Management

```python
from pyflame_rt.custom import CustomOpRegistry

registry = CustomOpRegistry.instance()

# List all custom operators
print(registry.list())

# Check if operator exists
if registry.has("MyGELU"):
    op = registry.get("MyGELU")
    print(f"Operator: {op.full_name()}")

# Unregister operator
registry.unregister("MyGELU")
```

### Graph Partitioning

Partition graphs across multiple devices for data parallel, model parallel, or pipeline parallel execution.

#### Quick Start

```python
from pyflame_rt import partition

# Auto-partition for 4 devices
session = pyflame_rt.InferenceSession("large_model.pfm")
graph = session.graph()

plan = partition.auto_partition(graph, num_devices=4)
print(f"Partitions: {len(plan.partitions)}")
print(f"Communication: {plan.total_comm_bytes / 1e6:.1f} MB")
```

#### Partition Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `DATA_PARALLEL` | Replicate model, split data | High throughput, model fits in memory |
| `MODEL_PARALLEL` | Split model across devices | Large models that don't fit in memory |
| `PIPELINE_PARALLEL` | Pipeline stages across devices | Very large models, overlapped compute |
| `AUTOMATIC` | Cost-model based selection | Let the system decide |

#### Data Parallel

```python
config = partition.PartitionConfig()
config.strategy = partition.PartitionStrategy.DATA_PARALLEL

for i in range(4):
    device = partition.DeviceSpec()
    device.type = partition.DeviceType.CPU
    device.device_id = i
    config.devices.append(device)

partitioner = partition.GraphPartitioner(config)
plan = partitioner.partition_data_parallel(graph, num_replicas=4)
```

#### Model Parallel

```python
config = partition.PartitionConfig()
config.strategy = partition.PartitionStrategy.MODEL_PARALLEL
config.balance_compute = True
config.minimize_communication = True

partitioner = partition.GraphPartitioner(config)
plan = partitioner.partition_model_parallel(graph, num_partitions=4)

# Check balance
stats = partition.GraphPartitioner.get_stats(plan)
print(f"Nodes per partition: {stats.nodes_per_partition}")
print(f"Edge cut ratio: {stats.edge_cut_ratio:.2%}")
```

#### Pipeline Parallel

```python
config = partition.PartitionConfig()
config.strategy = partition.PartitionStrategy.PIPELINE_PARALLEL
config.max_pipeline_stages = 4
config.micro_batch_size = 8

partitioner = partition.GraphPartitioner(config)
plan = partitioner.partition_pipeline(graph, num_stages=4)

print(f"Load imbalance: {plan.load_imbalance:.2%}")
```

#### WSE Multi-Chip Partitioning

Partition for Cerebras WSE multi-chip configurations:

```python
from pyflame_rt.partition import wse

# Configure 2x2 chip grid
chip_config = partition.WSEChipConfig()
chip_config.topology = [2, 2]
chip_config.inter_chip_bandwidth = 1e12  # 1 TB/s
chip_config.inter_chip_latency = 100.0   # 100 ns
chip_config.chip_memory_bytes = 40 * 1024**3  # 40 GB per chip

# Partition for WSE
plan = wse.partition_for_wse(graph, chip_config)

# Optimize for WSE dataflow
optimized_graph = wse.optimize_for_wse_dataflow(graph)
```

#### Cost Model

Customize the cost model for accurate partitioning:

```python
cost_model = partition.CostModel()
cost_model.set_inter_device_bandwidth(100e9)  # 100 GB/s
cost_model.set_inter_device_latency(1.0)      # 1 us

partitioner = partition.GraphPartitioner(config)
partitioner.set_cost_model(cost_model)
```

#### Executing Partitioned Graphs

```python
# Create executor
executor = partition.PartitionedExecutor(plan)

# Run inference
inputs = {"input": input_tensor}
outputs = executor.execute(inputs)

# Async execution
future = executor.execute_async(inputs)
outputs = future.result()
```

#### Graph Analysis

Analyze graphs before partitioning:

```python
partitioner = partition.GraphPartitioner(config)
analysis = partitioner.analyze(graph)

print(f"Recommended strategy: {analysis.recommended_strategy}")
print(f"Estimated speedup: {analysis.estimated_speedup:.2f}x")
print(f"Communication: {analysis.communication_bytes / 1e6:.1f} MB")
print(f"Bottleneck nodes: {analysis.bottleneck_nodes}")
```

### C++ API for Advanced Optimization

#### Pruning in C++

```cpp
#include "pyflame_rt/pruning/pruning.hpp"

using namespace pyflame_rt::pruning;

// Configure pruning
PruningConfig config = PruningConfig::magnitude_pruning(0.5f);
WeightPruner pruner(config);

// Prune graph
Graph pruned_graph = pruner.prune(original_graph);

// Get statistics
PruningStats stats = pruner.get_stats();
std::cout << "Sparsity: " << stats.actual_sparsity() << std::endl;
```

#### Distillation in C++

```cpp
#include "pyflame_rt/distillation/distillation.hpp"

using namespace pyflame_rt::distillation;

// Create trainer
DistillationConfig config = DistillationConfig::soft_label(4.0f);
DistillationTrainer trainer(teacher_graph, student_graph, config);

// Configure training
TrainingConfig train_config;
train_config.learning_rate = 1e-4f;
train_config.num_epochs = 10;
trainer.set_training_config(train_config);

// Train
DistillationResult result = trainer.train();
```

#### Custom Ops in C++

```cpp
#include "pyflame_rt/custom/custom_op.hpp"

using namespace pyflame_rt::custom;

// Register operator
CustomOp& op = CustomOpBuilder("MyOp")
    .domain("custom")
    .input("X", DType::Float32)
    .output("Y", DType::Float32)
    .kernel([](const std::vector<Tensor>& inputs,
               const std::unordered_map<std::string, std::any>& attrs) {
        // Implementation
        return std::vector<Tensor>{inputs[0].clone()};
    })
    .build();
```

#### Partitioning in C++

```cpp
#include "pyflame_rt/partition/partition.hpp"

using namespace pyflame_rt::partition;

// Configure partitioning
PartitionConfig config;
config.strategy = PartitionStrategy::PipelineParallel;
for (int i = 0; i < 4; ++i) {
    DeviceSpec device;
    device.type = DeviceType::CPU;
    device.device_id = i;
    config.devices.push_back(device);
}

// Partition graph
GraphPartitioner partitioner(config);
PartitionPlan plan = partitioner.partition(graph);

// Execute
PartitionedExecutor executor(plan);
auto outputs = executor.execute(inputs);
```

### Best Practices for Advanced Optimization

#### Pruning Best Practices

1. **Start with structured pruning** for hardware compatibility
2. **Use gradual pruning** for better accuracy retention
3. **Fine-tune after pruning** to recover accuracy
4. **Exclude sensitive layers** like final classifiers
5. **Validate accuracy** at each sparsity level

#### Distillation Best Practices

1. **Use larger temperatures** (4-20) for softer targets
2. **Combine with hard labels** for classification tasks
3. **Match intermediate features** for deep networks
4. **Use representative training data**
5. **Monitor validation loss** for early stopping

#### Custom Ops Best Practices

1. **Keep kernels simple** and well-tested
2. **Implement shape inference** for graph optimization
3. **Add gradient functions** if training is needed
4. **Use appropriate data types**
5. **Handle edge cases** (empty tensors, broadcasting)

#### Partitioning Best Practices

1. **Analyze before partitioning** to choose strategy
2. **Minimize communication** across partition boundaries
3. **Balance compute** across devices
4. **Use pipeline parallelism** for very large models
5. **Consider memory constraints** per device

---

## Python API

### InferenceSession

The primary interface for running inference:

```python
import numpy as np
import pyflame_rt

# Basic usage
session = pyflame_rt.InferenceSession("model.pfm")

# With options
options = pyflame_rt.SessionOptions()
options.device = "cpu"
options.num_threads = 4

session = pyflame_rt.InferenceSession(
    "model.pfm",
    options=options,
    providers=["CPUExecutionProvider"]
)
```

#### Session Methods

```python
# Get input/output metadata
inputs = session.get_inputs()
outputs = session.get_outputs()

for inp in inputs:
    print(f"Input '{inp.name}': shape={inp.shape}, type={inp.type}")

for out in outputs:
    print(f"Output '{out.name}': shape={out.shape}, type={out.type}")

# Get model metadata
meta = session.get_modelmeta()
print(f"Producer: {meta.producer_name} {meta.producer_version}")
print(f"Graph name: {meta.graph_name}")

# Get available providers
providers = session.get_providers()
print(f"Providers: {providers}")
```

#### Running Inference

```python
# Prepare input
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

# Run inference - all outputs
results = session.run(None, {"input": input_data})

# Run inference - specific outputs
results = session.run(["output"], {"input": input_data})

# Run with options
run_opts = pyflame_rt.RunOptions()
run_opts.tag = "inference_run_1"
results = session.run(None, {"input": input_data}, run_opts)

# Access results
for i, result in enumerate(results):
    print(f"Output {i}: shape={result.shape}")
```

### SessionOptions

Configure session behavior:

```python
options = pyflame_rt.SessionOptions()

# Device selection
options.device = "cpu"          # "cpu", "wse", "wse2", "wse3"

# Threading
options.num_threads = 0         # 0 = auto-detect

# Profiling
options.enable_profiling = True

# Execution mode
options.execution_mode = "sequential"  # "sequential" or "parallel"

# Optimization (Phase 3)
options.optimization_level = pyflame_rt.OptLevel.EXTENDED  # Default
options.verbose_optimization = False   # Log optimization details

# Logging
options.log_level = "warning"   # "debug", "info", "warning", "error"

# Validate options
errors = options.validate()
if errors:
    print(f"Invalid options: {errors}")
```

#### Optimization Levels

```python
# Available optimization levels
pyflame_rt.OptLevel.NONE      # No optimization (level 0)
pyflame_rt.OptLevel.BASIC     # Constant folding, DCE (level 1)
pyflame_rt.OptLevel.EXTENDED  # + CSE, Fusion (level 2, default)
pyflame_rt.OptLevel.ALL       # + Layout optimization (level 3)
```

### RunOptions

Per-inference configuration:

```python
run_opts = pyflame_rt.RunOptions()

# Override log level for this run
run_opts.log_level = "debug"

# Tag for profiling
run_opts.tag = "batch_1"

# Timeout
run_opts.timeout_ms = 5000  # 5 seconds
```

### Error Handling

PyFlameRT provides specific exception types:

```python
import pyflame_rt

try:
    session = pyflame_rt.InferenceSession("model.pfm")
except pyflame_rt.InvalidModelError as e:
    print(f"Invalid model: {e}")
except pyflame_rt.UnsupportedFormatError as e:
    print(f"Unsupported format: {e}")
except pyflame_rt.PyFlameRTError as e:
    print(f"Runtime error: {e}")

try:
    results = session.run(None, {"wrong_name": data})
except pyflame_rt.InputError as e:
    print(f"Input error: {e}")
except pyflame_rt.ShapeMismatchError as e:
    print(f"Shape mismatch: {e}")
except pyflame_rt.UnsupportedOperatorError as e:
    print(f"Unsupported op: {e}")
```

#### Exception Hierarchy

```
PyFlameRTError (base)
├── InvalidModelError      # Model file invalid/corrupted
├── UnsupportedFormatError # Unsupported file format
├── UnsupportedOperatorError # Op not supported by backend
├── ShapeMismatchError     # Input shape doesn't match
├── DTypeMismatchError     # Input dtype doesn't match
├── ValidationError        # Graph validation failed
├── BackendError           # Execution failed
└── InputError             # Invalid input provided
```

---

## C++ API

### Including Headers

```cpp
// Include everything
#include <pyflame_rt/pyflame_rt.hpp>

// Or include specific components
#include <pyflame_rt/tensor.hpp>
#include <pyflame_rt/graph.hpp>
#include <pyflame_rt/session.hpp>
```

### Tensor Class

```cpp
#include <pyflame_rt/tensor.hpp>

using namespace pyflame_rt;

// Create tensor with shape and dtype
Tensor tensor({2, 3, 4}, DType::Float32);

// Access properties
std::vector<int64_t> shape = tensor.shape();
DType dtype = tensor.dtype();
size_t ndim = tensor.ndim();
int64_t num_elements = tensor.num_elements();
size_t size_bytes = tensor.size_bytes();

// Access data
float* data = tensor.data_ptr<float>();
for (int64_t i = 0; i < tensor.num_elements(); ++i) {
    data[i] = static_cast<float>(i);
}

// Fill with value
tensor.fill(0.0f);

// Clone (deep copy)
Tensor clone = tensor.clone();

// Reshape
Tensor reshaped = tensor.reshape({6, 4});

// Create view (non-owning)
Tensor view = tensor.view();
```

### Node Class

```cpp
#include <pyflame_rt/node.hpp>

using namespace pyflame_rt;

// Create node
Node node(
    "relu_0",                          // name
    "Relu",                            // op_type
    {"input"},                         // inputs
    {"output"}                         // outputs
);

// Set attributes
node.set_attr("alpha", 0.01f);
node.set_attr("axis", int64_t(1));
node.set_attr("axes", std::vector<int64_t>{0, 2});

// Get attributes
float alpha = node.get_attr<float>("alpha", 0.0f);  // with default
auto axis = node.get_attr<int64_t>("axis");         // returns optional
```

### Graph Class

```cpp
#include <pyflame_rt/graph.hpp>

using namespace pyflame_rt;

// Create graph
Graph graph("my_model");

// Add inputs/outputs
TensorInfo input_info("input", {{1}, {3}, {224}, {224}}, DType::Float32);
TensorInfo output_info("output", {{1}, {1000}}, DType::Float32);
graph.add_input(input_info);
graph.add_output(output_info);

// Add initializers
Tensor weight({1000, 2048}, DType::Float32);
weight.fill(0.01f);
graph.add_initializer("fc_weight", std::move(weight));

// Add nodes
auto node = std::make_shared<Node>(
    "fc_0", "Gemm",
    std::vector<std::string>{"flatten_out", "fc_weight"},
    std::vector<std::string>{"output"}
);
graph.add_node(node);

// Validate graph
std::vector<std::string> errors = graph.validate();
if (!errors.empty()) {
    for (const auto& err : errors) {
        std::cerr << "Validation error: " << err << std::endl;
    }
}

// Topological sort
auto sorted_nodes = graph.topological_sort();
```

### InferenceSession (C++)

```cpp
#include <pyflame_rt/session.hpp>

using namespace pyflame_rt;

// Create session
SessionOptions options;
options.device = "cpu";
options.num_threads = 4;

InferenceSession session("model.pfm", options);

// Get metadata
auto inputs = session.get_inputs();
auto outputs = session.get_outputs();
auto meta = session.get_modelmeta();

// Run inference
std::unordered_map<std::string, Tensor> input_feed;
input_feed["input"] = std::move(input_tensor);

std::vector<Tensor> results = session.run({}, input_feed);
```

### Operator Registry

```cpp
#include <pyflame_rt/registry.hpp>

using namespace pyflame_rt;

// Get registry instance
auto& registry = OperatorRegistry::instance();

// Check if operator exists
if (registry.has("MyCustomOp")) {
    // Get operator function
    const OpFunc* op = registry.get("MyCustomOp");
}

// List all operators
std::vector<std::string> ops = registry.list_ops();

// Register custom operator
registry.register_op("MyCustomOp", [](
    const std::vector<const Tensor*>& inputs,
    const OpContext& ctx) -> std::vector<Tensor>
{
    // Implementation
    return {inputs[0]->clone()};
});
```

---

## Model Format

### PyFlame Model Format (.pfm)

PyFlameRT uses a binary format for model serialization:

```
┌──────────────────────────────────────────┐
│ Header                                    │
├──────────────────────────────────────────┤
│ Magic: "PFM\0" (4 bytes)                 │
│ Version: uint32 (4 bytes)                │
├──────────────────────────────────────────┤
│ Graph Name: string                        │
├──────────────────────────────────────────┤
│ Inputs: count + [TensorInfo...]          │
├──────────────────────────────────────────┤
│ Outputs: count + [TensorInfo...]         │
├──────────────────────────────────────────┤
│ Initializers: count + [name + Tensor...] │
├──────────────────────────────────────────┤
│ Nodes: count + [Node...]                 │
└──────────────────────────────────────────┘
```

### Creating Models Programmatically

```cpp
#include <pyflame_rt/graph.hpp>
#include "io/loader.hpp"

using namespace pyflame_rt;

// Build graph
Graph graph("simple_relu");

TensorInfo input("input", {{1}, {10}}, DType::Float32);
TensorInfo output("output", {{1}, {10}}, DType::Float32);
graph.add_input(input);
graph.add_output(output);

auto relu = std::make_shared<Node>(
    "relu_0", "Relu",
    std::vector<std::string>{"input"},
    std::vector<std::string>{"output"}
);
graph.add_node(relu);

// Save to file
save_model(graph, "simple_relu.pfm");
```

### Model Inspection

```python
import pyflame_rt

session = pyflame_rt.InferenceSession("model.pfm")

print("=== Model Information ===")
meta = session.get_modelmeta()
print(f"Graph: {meta.graph_name}")
print(f"Producer: {meta.producer_name} {meta.producer_version}")

print("\n=== Inputs ===")
for inp in session.get_inputs():
    print(f"  {inp.name}: {inp.shape} ({inp.type})")

print("\n=== Outputs ===")
for out in session.get_outputs():
    print(f"  {out.name}: {out.shape} ({out.type})")

print("\n=== Providers ===")
for provider in session.get_providers():
    print(f"  {provider}")
```

---

## Extending PyFlameRT

### Adding Custom Operators

#### C++ Implementation

```cpp
// src/backends/cpu/ops/custom.cpp

#include "pyflame_rt/registry.hpp"
#include "pyflame_rt/tensor.hpp"
#include <cmath>

namespace pyflame_rt {
namespace ops {

namespace {

// Custom GELU variant
std::vector<Tensor> cpu_custom_gelu(
    const std::vector<const Tensor*>& inputs,
    const OpContext& ctx)
{
    const Tensor& x = *inputs[0];
    Tensor result(x.shape(), x.dtype());

    const float* in = x.data_ptr<float>();
    float* out = result.data_ptr<float>();
    int64_t n = x.num_elements();

    // Get custom parameter
    float scale = ctx.node->get_attr<float>("scale", 1.0f);

    for (int64_t i = 0; i < n; ++i) {
        float val = in[i] * scale;
        // Approximate GELU
        out[i] = 0.5f * val * (1.0f + std::tanh(
            0.7978845608f * (val + 0.044715f * val * val * val)));
    }

    return {std::move(result)};
}

// Register at static initialization
struct CustomOpsRegistrar {
    CustomOpsRegistrar() {
        OperatorRegistry::instance().register_op("CustomGelu", cpu_custom_gelu);
    }
};

static CustomOpsRegistrar custom_ops_registrar;

} // anonymous namespace
} // namespace ops
} // namespace pyflame_rt
```

#### Adding to Build

Add the source file to `src/CMakeLists.txt`:

```cmake
set(PYFLAME_RT_SOURCES
    # ... existing sources ...
    backends/cpu/ops/custom.cpp   # Add this line
)
```

#### Using Custom Operators

```cpp
// In graph construction
auto node = std::make_shared<Node>(
    "custom_gelu_0", "CustomGelu",
    std::vector<std::string>{"input"},
    std::vector<std::string>{"output"}
);
node->set_attr("scale", 1.5f);
graph.add_node(node);
```

### Operator Implementation Guidelines

1. **Input Validation**: Check tensor count and shapes
2. **Attribute Access**: Use `get_attr` with defaults
3. **Memory Efficiency**: Minimize copies
4. **Numerical Stability**: Handle edge cases (overflow, NaN)
5. **Documentation**: Add to supported operators list

```cpp
std::vector<Tensor> cpu_my_op(
    const std::vector<const Tensor*>& inputs,
    const OpContext& ctx)
{
    // 1. Validate inputs
    if (inputs.size() < 2) {
        throw std::invalid_argument("MyOp requires at least 2 inputs");
    }

    const Tensor& a = *inputs[0];
    const Tensor& b = *inputs[1];

    // 2. Get attributes with defaults
    float alpha = ctx.node->get_attr<float>("alpha", 1.0f);
    int64_t axis = ctx.node->get_attr<int64_t>("axis", -1);

    // 3. Validate shapes
    if (a.shape() != b.shape()) {
        throw std::invalid_argument("Shape mismatch");
    }

    // 4. Allocate output
    Tensor result(a.shape(), a.dtype());

    // 5. Compute
    const float* a_data = a.data_ptr<float>();
    const float* b_data = b.data_ptr<float>();
    float* out_data = result.data_ptr<float>();

    for (int64_t i = 0; i < a.num_elements(); ++i) {
        out_data[i] = alpha * a_data[i] + b_data[i];
    }

    return {std::move(result)};
}
```

### Creating Custom Backends

For advanced use cases, you can implement custom backends:

```cpp
// include/mybackend/my_executor.hpp

#include <pyflame_rt/backend.hpp>

namespace mybackend {

class MyExecutor : public pyflame_rt::Backend {
public:
    MyExecutor(const Config& config);

    const std::string& name() const override { return name_; }

    bool supports_op(const std::string& op_type) const override;

    std::vector<std::string> get_supported_ops() const override;

    std::vector<pyflame_rt::Tensor> execute(
        const pyflame_rt::Graph& graph,
        const std::unordered_map<std::string, pyflame_rt::Tensor>& input_feed,
        const std::vector<std::string>& output_names = {}
    ) override;

private:
    std::string name_ = "my_backend";
    Config config_;
    // Backend-specific state
};

} // namespace mybackend
```

---

## Performance Considerations

### Memory Management

```cpp
// Good: Move semantics
Tensor result = compute_something();
return result;  // Moved, not copied

// Good: Avoid unnecessary copies
const Tensor& input = *inputs[0];  // Reference, no copy

// Avoid: Unnecessary cloning
Tensor copy = input.clone();  // Only when needed
```

### Tensor Views

```cpp
// Views share underlying data
Tensor original({1000, 1000}, DType::Float32);
Tensor view = original.view();  // No memory allocation

// Modifications affect both
view.data_ptr<float>()[0] = 42.0f;
assert(original.data_ptr<float>()[0] == 42.0f);
```

### Threading

```python
# CPU threading for parallel operators
options = pyflame_rt.SessionOptions()
options.num_threads = 8  # Use 8 threads

# 0 means auto-detect (usually num_cores)
options.num_threads = 0
```

### Batch Processing

```python
# Process larger batches for better throughput
batch_sizes = [1, 4, 16, 32]

for batch_size in batch_sizes:
    input_data = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)
    results = session.run(None, {"input": input_data})
```

### Profiling

```python
options = pyflame_rt.SessionOptions()
options.enable_profiling = True

session = pyflame_rt.InferenceSession("model.pfm", options)

# Run with profiling tag
run_opts = pyflame_rt.RunOptions()
run_opts.tag = "warmup"
session.run(None, inputs, run_opts)

run_opts.tag = "inference"
for i in range(100):
    session.run(None, inputs, run_opts)

# Profiling data accessible through backend (future)
```

---

## Debugging and Troubleshooting

### Common Issues

#### 1. Model Loading Failures

```python
# Check file exists and format
import os

model_path = "model.pfm"
if not os.path.exists(model_path):
    print("Model file not found")
elif not model_path.endswith('.pfm'):
    print("Unsupported format - use .pfm")
else:
    try:
        session = pyflame_rt.InferenceSession(model_path)
    except pyflame_rt.InvalidModelError as e:
        print(f"Invalid model: {e}")
```

#### 2. Shape Mismatches

```python
# Check expected vs actual shapes
inputs = session.get_inputs()
for inp in inputs:
    print(f"Expected: {inp.name} = {inp.shape}")

# Verify your input
print(f"Actual: input = {input_data.shape}")

# Handle dynamic batch
if inputs[0].shape[0] is None:
    print("Batch dimension is dynamic")
```

#### 3. Unsupported Operators

```python
try:
    session = pyflame_rt.InferenceSession("model.pfm")
except pyflame_rt.UnsupportedOperatorError as e:
    print(f"Operator not supported: {e}")
    # Check available operators
    # (Would need to expose from registry)
```

#### 4. Memory Issues

```python
# For large models, process in smaller batches
def process_in_batches(session, data, batch_size=32):
    results = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        result = session.run(None, {"input": batch})
        results.append(result[0])
    return np.concatenate(results)
```

### Logging

```python
options = pyflame_rt.SessionOptions()
options.log_level = "debug"  # Maximum verbosity

session = pyflame_rt.InferenceSession("model.pfm", options)
```

### Graph Validation

```python
# Validation happens automatically on load
try:
    session = pyflame_rt.InferenceSession("model.pfm")
except pyflame_rt.ValidationError as e:
    print(f"Graph validation failed:")
    for error in e.errors:
        print(f"  - {error}")
```

---

## Best Practices

### 1. Session Reuse

```python
# Good: Create session once, reuse
session = pyflame_rt.InferenceSession("model.pfm")
for batch in data_loader:
    results = session.run(None, {"input": batch})

# Avoid: Creating session repeatedly
for batch in data_loader:
    session = pyflame_rt.InferenceSession("model.pfm")  # Expensive!
    results = session.run(None, {"input": batch})
```

### 2. Input Preparation

```python
import numpy as np

# Good: Ensure correct dtype upfront
input_data = data.astype(np.float32)

# Good: Ensure contiguous memory
input_data = np.ascontiguousarray(input_data)

# Good: Match expected shape
if len(input_data.shape) == 3:
    input_data = input_data[np.newaxis, ...]  # Add batch dim
```

### 3. Error Handling

```python
def safe_inference(session, inputs):
    """Run inference with proper error handling."""
    try:
        return session.run(None, inputs)
    except pyflame_rt.InputError as e:
        logger.error(f"Input error: {e}")
        raise
    except pyflame_rt.BackendError as e:
        logger.error(f"Backend error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise
```

### 4. Resource Cleanup

```python
# Sessions are automatically cleaned up when out of scope
def process_model(path, data):
    session = pyflame_rt.InferenceSession(path)
    return session.run(None, {"input": data})
# Session is released here

# For explicit control, use context managers (future feature)
```

### 5. Type Safety

```python
# Check types match
input_info = session.get_inputs()[0]
expected_type = input_info.type  # e.g., "tensor(float32)"

if "float32" in expected_type:
    data = data.astype(np.float32)
elif "float16" in expected_type:
    data = data.astype(np.float16)
```

### 6. Testing

```python
import numpy as np
import pyflame_rt

def test_model_outputs():
    """Verify model produces expected outputs."""
    session = pyflame_rt.InferenceSession("model.pfm")

    # Test with known input
    test_input = np.ones((1, 3, 224, 224), dtype=np.float32)
    results = session.run(None, {"input": test_input})

    # Verify output shape
    assert results[0].shape == (1, 1000)

    # Verify output is valid
    assert not np.isnan(results[0]).any()
    assert not np.isinf(results[0]).any()

    # Verify probabilities sum to ~1 for classification
    probs = np.exp(results[0]) / np.exp(results[0]).sum()
    assert np.isclose(probs.sum(), 1.0, atol=1e-5)
```

---

## Appendix A: Supported Operators

### Math Operators

| Operator | Description | Attributes |
|----------|-------------|------------|
| `Add` | Element-wise addition | - |
| `Sub` | Element-wise subtraction | - |
| `Mul` | Element-wise multiplication | - |
| `Div` | Element-wise division | - |
| `MatMul` | Matrix multiplication | - |
| `Gemm` | General matrix multiply | `alpha`, `beta`, `transA`, `transB` |
| `Sqrt` | Element-wise square root | - |
| `Exp` | Element-wise exponential | - |
| `Log` | Element-wise natural log | - |
| `Pow` | Element-wise power | - |
| `Neg` | Element-wise negation | - |
| `Abs` | Element-wise absolute value | - |

### Activation Operators

| Operator | Description | Attributes |
|----------|-------------|------------|
| `Relu` | ReLU activation | - |
| `Sigmoid` | Sigmoid activation | - |
| `Tanh` | Hyperbolic tangent | - |
| `LeakyRelu` | Leaky ReLU | `alpha` (default: 0.01) |
| `Elu` | ELU activation | `alpha` (default: 1.0) |
| `Selu` | SELU activation | - |
| `Softmax` | Softmax | `axis` (default: -1) |
| `Gelu` | GELU activation | - |
| `HardSwish` | Hard Swish | - |

### Tensor Operators

| Operator | Description | Attributes |
|----------|-------------|------------|
| `Reshape` | Change tensor shape | `shape` |
| `Transpose` | Permute dimensions | `perm` |
| `Concat` | Concatenate tensors | `axis` |
| `Squeeze` | Remove dimensions | `axes` |
| `Unsqueeze` | Add dimensions | `axes` |
| `Flatten` | Flatten to 2D | `axis` (default: 1) |
| `Slice` | Extract subtensor | `starts`, `ends`, `axes`, `steps` |
| `Identity` | Copy tensor | - |
| `Constant` | Create constant tensor | `value`, `shape` |

### Reduction Operators

| Operator | Description | Attributes |
|----------|-------------|------------|
| `ReduceSum` | Sum reduction | `axes`, `keepdims` |
| `ReduceMean` | Mean reduction | `axes`, `keepdims` |
| `ReduceMax` | Max reduction | `axes`, `keepdims` |
| `ReduceMin` | Min reduction | `axes`, `keepdims` |
| `ArgMax` | Argmax | `axis`, `keepdims` |

### Neural Network Operators

| Operator | Description | Attributes |
|----------|-------------|------------|
| `Conv` | Convolution | `kernel_shape`, `strides`, `pads`, `dilations`, `group` |
| `MaxPool` | Max pooling | `kernel_shape`, `strides`, `pads` |
| `AveragePool` | Average pooling | `kernel_shape`, `strides`, `pads` |
| `GlobalAveragePool` | Global average pooling | - |
| `BatchNormalization` | Batch normalization | `epsilon` |
| `LayerNormalization` | Layer normalization | `axis`, `epsilon` |
| `Dropout` | Dropout (identity in inference) | - |

---

## Appendix B: Version History

### Version 0.7.0 (Phase 7)

**Advanced Optimization Release**

- Comprehensive weight pruning for model compression
- Knowledge distillation for training smaller models
- Custom operator registration system
- Graph partitioning for multi-device execution

**New Features:**
- **Weight Pruning**: Unstructured, structured, block, and N:M sparsity patterns
- **Pruning Criteria**: Magnitude, movement, Taylor, LAMP importance scoring
- **Pruning Schedules**: One-shot, iterative, cubic, polynomial schedules
- **Sparse Tensors**: COO, CSR, CSC, BSR storage formats with sparse operations
- **Knowledge Distillation**: Soft label, feature, and attention transfer
- **Student Architecture**: Automatic generation of compressed student models
- **Training Support**: LR scheduling, early stopping, checkpointing
- **Custom Operators**: Fluent builder API for operator registration
- **Shape/Type Inference**: Automatic inference for custom operators
- **Gradient Support**: Gradient functions for custom operator training
- **Graph Partitioning**: Data, model, and pipeline parallelism
- **Cost Model**: Compute and communication cost estimation
- **WSE Multi-Chip**: Optimized partitioning for Cerebras WSE topology

**API Additions:**
- `pyflame_rt.pruning` module - Weight pruning submodule
- `pyflame_rt.pruning.PruningConfig` - Pruning configuration
- `pyflame_rt.pruning.PruningGranularity` - Unstructured, Structured, Block, NM
- `pyflame_rt.pruning.PruningCriterion` - Magnitude, Movement, Taylor, LAMP
- `pyflame_rt.pruning.PruningSchedule` - OneShot, Iterative, Cubic, Polynomial
- `pyflame_rt.pruning.WeightPruner` - Main pruning class
- `pyflame_rt.pruning.PruningMask` - Binary pruning masks
- `pyflame_rt.pruning.SparseTensor` - Sparse tensor representation
- `pyflame_rt.distillation` module - Knowledge distillation submodule
- `pyflame_rt.distillation.DistillationConfig` - Distillation configuration
- `pyflame_rt.distillation.DistillationLoss` - KLDivergence, MSE, Cosine, Attention
- `pyflame_rt.distillation.StudentConfig` - Student model configuration
- `pyflame_rt.distillation.TrainingConfig` - Training configuration
- `pyflame_rt.distillation.DistillationTrainer` - Distillation training
- `pyflame_rt.distillation.InMemoryDataset` - Dataset for training
- `pyflame_rt.custom` module - Custom operator submodule
- `pyflame_rt.custom.CustomOpBuilder` - Fluent operator builder
- `pyflame_rt.custom.CustomOpRegistry` - Operator registry
- `pyflame_rt.custom.CustomOp` - Custom operator instance
- `pyflame_rt.partition` module - Graph partitioning submodule
- `pyflame_rt.partition.PartitionConfig` - Partition configuration
- `pyflame_rt.partition.PartitionStrategy` - DataParallel, ModelParallel, Pipeline
- `pyflame_rt.partition.GraphPartitioner` - Graph partitioning
- `pyflame_rt.partition.PartitionedExecutor` - Multi-device execution
- `pyflame_rt.partition.CostModel` - Partitioning cost model
- `pyflame_rt.partition.wse` - WSE-specific utilities

**Known Limitations:**
- CPU backend only for sparse tensor operations
- Distillation requires manual training loop integration
- Pipeline parallelism communication not overlapped with compute

---

### Version 0.4.0 (Phase 4)

**Quantization Release**

- Comprehensive quantization support for model compression
- Half-precision types: FP16 (IEEE 754) and BFloat16
- INT8 quantization with dynamic and static modes
- Calibration system for static quantization

**New Features:**
- **FP16 Quantization**: 2x memory reduction with minimal accuracy loss
- **BFloat16 Quantization**: Training-friendly format with FP32 dynamic range
- **Dynamic INT8**: Runtime quantization without calibration data
- **Static INT8**: Calibration-based quantization for best accuracy
- **Calibration Methods**: MinMax, Entropy (KL-divergence), Percentile
- **Per-Channel Quantization**: More accurate quantization for CNNs
- **Quantized Operators**: QuantizedMatMul, QuantizedAdd, and more
- Automatic quantization during session creation
- Manual quantization API for advanced control

**API Additions:**
- `pyflame_rt.quantization` module - Quantization submodule
- `pyflame_rt.quantization.QuantConfig` - Quantization configuration
- `pyflame_rt.quantization.QuantMode` - FP16, BFloat16, DynamicInt8, StaticInt8
- `pyflame_rt.quantization.QuantGranularity` - PerTensor, PerChannel
- `pyflame_rt.quantization.CalibrationMethod` - MinMax, Entropy, Percentile
- `pyflame_rt.quantization.QuantParams` - Scale and zero-point parameters
- `pyflame_rt.quantization.GraphQuantInfo` - Graph-level quantization info
- `pyflame_rt.quantization.Quantizer` - Graph quantization transformer
- `pyflame_rt.quantization.Calibrator` - Calibration data collector
- `pyflame_rt.quantization.QuantizationResult` - Quantization result with stats
- `SessionOptions.quantization` - Enable automatic quantization
- `SessionOptions.calibration_data` - Calibration data provider
- `InferenceSession.is_quantized()` - Check if session uses quantization
- `InferenceSession.quantization_report()` - Get quantization statistics

**Known Limitations:**
- CPU backend only (Cerebras backends in future phases)
- Some operators excluded from INT8 (Softmax, LayerNorm)
- Per-channel quantization slightly slower than per-tensor

---

### Version 0.3.0 (Phase 3)

**Graph Optimization Release**

- Comprehensive graph optimization framework
- Pass manager with dependency ordering and fixed-point iteration
- Pattern matcher for declarative fusion rules
- Five built-in optimization passes

**New Features:**
- **Constant Folding**: Pre-compute operations with static inputs
- **Dead Code Elimination**: Remove unused nodes and initializers
- **Common Subexpression Elimination**: Share identical computations
- **Operator Fusion**: Combine operations (Conv+BN+ReLU, MatMul+Add, etc.)
- **Layout Optimization**: Optimize tensor layouts (NCHW/NHWC)
- Automatic optimization during session creation
- Configurable optimization levels (None, Basic, Extended, All)
- Pattern-based fusion with single-consumer constraints

**API Additions:**
- `pyflame_rt.opt` module - Optimization submodule
- `pyflame_rt.opt.PassManager` - Pass orchestration
- `pyflame_rt.opt.PassManagerConfig` - Pass manager configuration
- `pyflame_rt.opt.PassStats` / `PassResult` - Optimization statistics
- `pyflame_rt.opt.optimize()` - Quick optimization function
- `pyflame_rt.opt.fold_constants()` - Manual constant folding
- `pyflame_rt.opt.eliminate_dead_code()` - Manual DCE
- `pyflame_rt.opt.eliminate_common_subexpressions()` - Manual CSE
- `pyflame_rt.opt.fuse_operators()` - Manual fusion
- Individual pass configs: `ConstantFoldingConfig`, `DCEConfig`, `CSEConfig`, `FusionConfig`, `LayoutConfig`
- `SessionOptions.optimization_level` - Control optimization level
- `SessionOptions.verbose_optimization` - Enable optimization logging

**Known Limitations:**
- CPU backend only (Cerebras backends in future phases)
- No quantization support (Phase 4)
- Layout optimization conservative (skips if layouts consistent)
- Some fusion patterns require backend support for fused ops

---

### Version 0.2.0 (Phase 2)

**Model Import Release**

- ONNX model importer with opset 9-21 support
- PyTorch state dict importer with user-defined graph
- TorchScript model importer (traced and scripted)
- Automatic shape inference engine
- Importer registry with auto-registration

**New Features:**
- `from_onnx()` convenience function for quick ONNX import
- `ImportOptions` for configuring import behavior
- `ImportResult` with statistics and diagnostics
- 50+ ONNX operator converters
- Python pickle parser for PyTorch checkpoints
- ZIP archive extraction for modern model formats
- Shape inference for 30+ operators

**API Additions:**
- `pyflame_rt.from_onnx(path)` - Quick ONNX import
- `pyflame_rt.import_module.ONNXImporter` - Full ONNX importer
- `pyflame_rt.import_module.PyTorchImporter` - PyTorch importer
- `pyflame_rt.import_module.TorchScriptImporter` - TorchScript importer
- `pyflame_rt.ImportOptions` - Import configuration
- `pyflame_rt.ImportResult` - Import result with stats

**Known Limitations:**
- CPU backend only (Cerebras backends in future phases)
- ~~No graph optimization passes (Phase 3)~~ *Added in 0.3.0*
- No quantization support (Phase 4)
- PyTorch import requires user-defined graph structure

---

### Version 0.1.0 (Phase 1)

**Initial Release**

- Core C++ library with Python bindings
- CPU reference backend with 50+ operators
- Binary model format (.pfm)
- ONNX Runtime-compatible InferenceSession API
- Comprehensive test suite

**Supported Features:**
- Tensor operations with memory management
- Graph IR with validation and topological sort
- Operator registry system
- SessionOptions and RunOptions
- NumPy integration

---

*Documentation generated for PyFlameRT v0.7.0*
