# PyFlameRT - Project Guide for Claude

This document provides a comprehensive overview of the PyFlameRT project for AI assistants and developers.

---

## Project Overview

**PyFlameRT** (PyFlame Runtime) is a high-performance inference runtime designed for Cerebras Wafer-Scale Engine (WSE) hardware, with CPU backend support. It provides a TensorRT-like interface for efficient neural network inference, similar to how NVIDIA's TensorRT optimizes inference for NVIDIA GPUs.

### Key Information

- **Language**: C++17 with Python bindings (pybind11)
- **Build System**: CMake 3.18+
- **Testing**: GoogleTest (C++), pytest (Python)
- **Python Version**: 3.8+
- **Status**: Phase 3 - Graph Optimization (Complete)

### Project Goals

1. Provide a high-performance inference runtime for Cerebras WSE
2. Support CPU execution for development and testing
3. Offer ONNX Runtime-compatible API for ease of adoption
4. Enable custom operator and backend extensions
5. Support dynamic batch sizes and graph optimization

---

## Project Structure

```
PyFlameRT/
├── CMakeLists.txt              # Root CMake configuration
├── README.md                   # User-facing documentation
├── Project_Overview.md         # High-level project vision
├── Phase1_Architecture.md      # Phase 1 architecture details
├── Phase1_Implementation_Plan.md # Phase 1 implementation plan
│
├── include/pyflame_rt/         # Public C++ headers
│   ├── types.hpp               # Core type definitions (DType, Shape, TensorInfo)
│   ├── errors.hpp              # Exception hierarchy
│   ├── tensor.hpp              # Tensor class (multi-dimensional arrays)
│   ├── node.hpp                # Node class (graph operations)
│   ├── graph.hpp               # Graph class (computation graph)
│   ├── registry.hpp            # OperatorRegistry (singleton)
│   ├── backend.hpp             # Backend interface
│   ├── options.hpp             # SessionOptions, RunOptions, CompileOptions
│   ├── session.hpp             # InferenceSession (main API)
│   └── opt/                    # Graph optimization (Phase 3)
│       ├── pass.hpp            # Pass manager infrastructure
│       ├── pattern_matcher.hpp # Graph pattern matching
│       ├── constant_folding.hpp # Constant folding pass
│       ├── dead_code_elimination.hpp # DCE pass
│       ├── cse.hpp             # Common subexpression elimination
│       ├── operator_fusion.hpp # Operator fusion pass
│       ├── layout_optimization.hpp # Layout optimization
│       └── passes.hpp          # Master include header
│
├── src/                        # C++ implementation
│   ├── CMakeLists.txt
│   ├── types.cpp               # Type utilities
│   ├── errors.cpp              # Exception implementations
│   ├── tensor.cpp              # Tensor implementation
│   ├── node.cpp                # Node implementation
│   ├── graph.cpp               # Graph implementation
│   ├── registry.cpp            # Operator registry
│   ├── options.cpp             # Options validation
│   ├── session.cpp             # InferenceSession implementation
│   │
│   ├── backends/cpu/           # CPU backend
│   │   ├── executor.cpp        # CPU graph executor
│   │   └── ops/                # CPU operator implementations
│   │       ├── math.cpp        # Add, Sub, Mul, Div, MatMul, Gemm
│   │       ├── activation.cpp  # Relu, Sigmoid, Tanh, Softmax
│   │       ├── tensor_ops.cpp  # Reshape, Transpose, Concat, Split
│   │       ├── reduction.cpp   # ReduceSum, ReduceMean, ReduceMax
│   │       └── nn.cpp          # Conv, MaxPool, BatchNorm, Dropout
│   │
│   ├── io/                     # Model I/O
│   │   ├── model_io.cpp        # Model serialization/deserialization
│   │   └── pfm_format.cpp      # .pfm format handler
│   │
│   └── opt/                    # Graph optimization (Phase 3)
│       ├── CMakeLists.txt      # Build configuration
│       ├── pass_manager.cpp    # PassManager implementation
│       ├── pattern_matcher.cpp # Pattern matching implementation
│       ├── constant_folding.cpp # Constant folding pass
│       ├── dead_code_elimination.cpp # DCE pass
│       ├── cse.cpp             # CSE pass
│       ├── operator_fusion.cpp # Operator fusion
│       ├── fusion_patterns.cpp # Built-in fusion patterns
│       └── layout_optimization.cpp # Layout optimization
│
├── bindings/                   # Python bindings (pybind11)
│   ├── CMakeLists.txt
│   ├── pyflame_rt.cpp          # Main Python module
│   ├── tensor_bind.cpp         # Tensor bindings + NumPy conversion
│   ├── types_bind.cpp          # Types and options bindings
│   ├── session_bind.cpp        # InferenceSession bindings
│   ├── import_bind.cpp         # Model import bindings
│   └── opt_bind.cpp            # Optimization bindings (Phase 3)
│
├── python/pyflame_rt/          # Python package
│   └── __init__.py             # Public API exports
│
├── tests/                      # Tests
│   ├── CMakeLists.txt
│   ├── cpp/                    # C++ unit tests (GoogleTest)
│   │   ├── CMakeLists.txt
│   │   ├── test_tensor.cpp
│   │   ├── test_graph.cpp
│   │   ├── test_registry.cpp
│   │   ├── test_cpu_ops.cpp
│   │   ├── test_model_io.cpp
│   │   ├── test_session.cpp
│   │   └── test_integration.cpp
│   │
│   └── python/                 # Python tests (pytest)
│       ├── test_bindings.py    # Test Python bindings
│       └── test_session.py     # Test InferenceSession
│
├── docs/                       # Documentation
│   ├── Developer_Guide.md      # Comprehensive developer guide
│   ├── API_Reference.md        # Complete API reference
│   ├── Extending_PyFlameRT.md  # Extension guide
│   └── Examples_and_Tutorials.md # Practical examples
│
├── third_party/                # Third-party dependencies
│   └── CMakeLists.txt          # GoogleTest, pybind11, Eigen (optional)
│
├── pyproject.toml              # Python package configuration
└── .gitignore                  # Git ignore rules
```

---

## Core Architecture

### 1. Type System

**File**: `include/pyflame_rt/types.hpp`

```cpp
enum class DType : uint8_t {
    Float32, Float16, BFloat16, Float64,
    Int64, Int32, Int16, Int8, UInt8, Bool
};

using Shape = std::vector<std::optional<int64_t>>;  // Dynamic dimensions = nullopt

struct TensorInfo {
    std::string name;
    DType dtype;
    Shape shape;
};

struct NodeArg {
    std::string name;
    DType dtype;
    Shape shape;
    std::string type_str() const;  // e.g., "tensor(float)"
};
```

### 2. Tensor Class

**File**: `include/pyflame_rt/tensor.hpp`

Multi-dimensional array with ownership semantics:
- **Owned tensors**: Allocate and manage their own memory
- **Borrowed tensors**: Reference external memory (views)
- **Copy-on-write**: Clone creates deep copy, view creates shallow reference
- **Type-safe data access**: `data_ptr<T>()` for typed access

### 3. Graph IR

**Files**: `include/pyflame_rt/{node.hpp, graph.hpp}`

- **Node**: Represents a single operation (op_type, inputs, outputs, attributes)
- **Graph**: Container for nodes, inputs, outputs, and initializers (weights)
- **Topological sort**: Ensures correct execution order
- **Validation**: Checks for missing inputs, undefined tensors, cycles

### 4. Operator Registry

**File**: `include/pyflame_rt/registry.hpp`

Singleton pattern for operator registration:
```cpp
using OpFunction = std::function<Tensor(
    const std::vector<Tensor>& inputs,
    const std::unordered_map<std::string, std::any>& attributes)>;

OperatorRegistry::instance().register_op("Add", cpu_add);
```

Operators are registered at static initialization time.

### 5. Backend Interface

**File**: `include/pyflame_rt/backend.hpp`

Abstract interface for execution backends:
- `initialize()` / `shutdown()`: Lifecycle management
- `execute()`: Run graph with inputs
- `allocate()` / `deallocate()`: Memory management
- `supports_op()`: Check operator support

Current backends:
- **CPUExecutionProvider**: CPU backend with 50+ operators

### 6. Inference Session

**File**: `include/pyflame_rt/session.hpp`

Main API for running inference:
```cpp
InferenceSession session("model.pfm");
auto outputs = session.run(
    {},  // Output names (empty = all)
    {{"input", input_tensor}}  // Input feed
);
```

ONNX Runtime-compatible API for easy migration.

---

## Key Design Decisions

### 1. Memory Management

- **Smart pointers**: Use `std::shared_ptr` for graph nodes
- **Ownership semantics**: Tensors track whether they own data
- **RAII**: All resources managed via destructors
- **No manual memory**: Avoid `new`/`delete` in favor of smart pointers

### 2. Error Handling

- **Exception hierarchy**: `PyFlameRTError` base class
  - `InvalidModelError`: Bad model files
  - `UnsupportedOperatorError`: Missing operators
  - `ValidationError`: Input/graph validation failures
- **Python integration**: Exceptions automatically converted to Python exceptions

### 3. Python Bindings

- **pybind11**: Modern C++/Python binding library
- **NumPy integration**: Zero-copy when possible, automatic conversion
- **Type mapping**: C++ types map naturally to Python types

### 4. Build System

- **CMake**: Cross-platform build system
- **Options**: `PYFLAME_RT_BUILD_TESTS`, `PYFLAME_RT_BUILD_PYTHON`
- **Dependencies**: GoogleTest, pybind11, optional Eigen

---

## Supported Operators (50+)

### Math Operators
Add, Sub, Mul, Div, Neg, Abs, Sqrt, Exp, Log, Pow, MatMul, Gemm

### Activation Functions
Relu, LeakyRelu, PRelu, Elu, Selu, Sigmoid, Tanh, Softmax, Softplus, Softsign

### Tensor Operations
Reshape, Transpose, Concat, Split, Slice, Squeeze, Unsqueeze, Flatten, Expand

### Reduction Operations
ReduceSum, ReduceMean, ReduceMax, ReduceMin, ReduceProd, ArgMax, ArgMin

### Neural Network Layers
Conv, MaxPool, AveragePool, GlobalAveragePool, BatchNormalization, Dropout, Gemm (FC)

---

## Model Format (.pfm)

Binary format for efficient model storage:

```
Header:
- Magic: "PFMODEL\0" (8 bytes)
- Version: uint32_t
- Metadata size: uint32_t

Metadata:
- Producer name, version, domain, description (JSON)

Graph:
- Name
- Inputs: count + TensorInfo[]
- Outputs: count + TensorInfo[]
- Nodes: count + Node[] (op_type, name, inputs, outputs, attributes)
- Initializers: count + (name, shape, dtype, data)[]
```

---

## Development Workflow

### Building the Project

```bash
# Configure
mkdir build && cd build
cmake .. -DPYFLAME_RT_BUILD_TESTS=ON -DPYFLAME_RT_BUILD_PYTHON=ON

# Build
cmake --build . --config Release

# Run C++ tests
ctest -C Release --output-on-failure

# Install Python package
pip install -e .

# Run Python tests
pytest ../tests/python/
```

### Adding a New Operator

1. **Implement operator function** in `src/backends/cpu/ops/*.cpp`
2. **Register operator** in static initializer
3. **Add tests** in `tests/cpp/test_cpu_ops.cpp`
4. **Update documentation** in `docs/API_Reference.md`

Example:
```cpp
// src/backends/cpu/ops/custom.cpp
Tensor cpu_my_op(const std::vector<Tensor>& inputs,
                 const std::unordered_map<std::string, std::any>& attrs) {
    // Implementation...
}

struct MyOpRegistrar {
    MyOpRegistrar() {
        OperatorRegistry::instance().register_op("MyOp", cpu_my_op);
    }
};
static MyOpRegistrar my_op_registrar;
```

### Adding a New Backend

1. **Implement Backend interface** in `include/pyflame_rt/backends/my_backend.hpp`
2. **Implement operators** for your backend
3. **Register backend** in backend registry
4. **Add tests**

---

## Graph Optimization (Phase 3)

PyFlameRT includes a comprehensive graph optimization framework:

### Pass Manager
- Orchestrates optimization passes with dependency ordering
- Fixed-point iteration until no more changes
- Configurable optimization levels (None, Basic, Extended, All)

### Optimization Passes

| Pass | Description |
|------|-------------|
| **Constant Folding** | Pre-compute operations with static inputs |
| **Dead Code Elimination** | Remove unused nodes and initializers |
| **CSE** | Common Subexpression Elimination - share identical computations |
| **Operator Fusion** | Combine operations (Conv+BN+ReLU, MatMul+Add→Gemm) |
| **Layout Optimization** | Optimize tensor layouts (NCHW/NHWC) |

### Pattern Matcher
- Declarative graph pattern matching for fusion
- Single-consumer constraint checking
- Flexible pattern definition

### Usage

```python
import pyflame_rt

# Automatic optimization (default: Extended level)
session = pyflame_rt.InferenceSession("model.pfm")

# Disable optimization
options = pyflame_rt.SessionOptions()
options.optimization_level = pyflame_rt.OptLevel.NONE

# Manual optimization
from pyflame_rt import opt
result = opt.optimize(graph, pyflame_rt.OptLevel.ALL)
```

---

## Future Phases

### Phase 2: Model Import (Complete)
- ONNX importer ✓
- PyTorch model converter ✓
- TorchScript importer ✓

### Phase 3: Graph Optimization (Complete)
- Operator fusion ✓
- Constant folding ✓
- Dead code elimination ✓
- Layout optimization ✓
- Common subexpression elimination ✓

### Phase 4: WSE Backend (Planned)
- Cerebras WSE integration
- WSE-specific optimizations
- Multi-chip support

### Phase 5: Quantization (Planned)
- INT8 quantization support
- Calibration tools
- Mixed-precision inference

---

## Testing Strategy

### C++ Tests (GoogleTest)
- **Unit tests**: Test individual classes (Tensor, Node, Graph)
- **Operator tests**: Test each operator implementation
- **Integration tests**: Test full inference pipeline

### Python Tests (pytest)
- **Binding tests**: Verify Python API works correctly
- **Session tests**: End-to-end inference tests
- **NumPy integration**: Test data conversion

### Coverage Goals
- C++ code: >80% line coverage
- Python bindings: >90% coverage
- All public APIs: 100% coverage

---

## Common Issues and Solutions

### Issue: "Cannot find pybind11"
**Solution**: Install pybind11 via pip or add to third_party/

### Issue: "Undefined symbols" when linking
**Solution**: Ensure all operators are registered in their respective .cpp files

### Issue: NumPy dtype mismatch
**Solution**: Explicitly convert NumPy arrays to float32 before passing to PyFlameRT

### Issue: Segfault in tensor operations
**Solution**: Check that tensor shapes are compatible, use validation functions

---

## Code Style and Conventions

### C++ Style
- **Naming**: `snake_case` for functions/variables, `PascalCase` for classes
- **Pointers**: Use `std::shared_ptr` and `std::unique_ptr`, avoid raw pointers
- **Const-correctness**: Mark methods const when they don't modify state
- **Headers**: Include guards via `#pragma once`

### Python Style
- **PEP 8**: Follow Python style guide
- **Type hints**: Use when appropriate (optional in this project)
- **Docstrings**: Document all public APIs

### CMake Style
- **Modern CMake**: Use targets and properties
- **Options**: Prefix with `PYFLAME_RT_`
- **Dependencies**: Use `find_package()` when possible

---

## Performance Considerations

### CPU Backend Performance
- **Multi-threading**: Use `SessionOptions::num_threads` to control threading
- **SIMD**: CPU operators should use vectorization where possible
- **Memory layout**: Tensors use contiguous memory (C-order)
- **Cache efficiency**: Operators minimize cache misses

### Memory Usage
- **Lazy allocation**: Only allocate when needed
- **Buffer reuse**: Reuse intermediate buffers when possible
- **Memory limit**: Set `SessionOptions::memory_limit` to cap usage

---

## Security Considerations

### Input Validation
- **Shape validation**: Check tensor shapes before operations
- **Type validation**: Verify dtypes match expectations
- **Bounds checking**: Array access should be bounds-checked

### Model Loading
- **Format validation**: Verify model file format before parsing
- **Size limits**: Prevent loading excessively large models
- **Malformed data**: Handle corrupted model files gracefully

### Buffer Management
- **Overflow prevention**: Check sizes before memory operations
- **Use std::vector**: Avoid fixed-size buffers when possible
- **Safe casting**: Use static_cast, avoid C-style casts

---

## Dependencies

### Required
- **C++17 compiler**: GCC 7+, Clang 5+, MSVC 2017+
- **CMake**: 3.18 or later
- **Python**: 3.8+ (for bindings)

### Optional
- **GoogleTest**: For C++ tests (auto-downloaded if not found)
- **pybind11**: For Python bindings (auto-downloaded if not found)
- **Eigen**: For optimized linear algebra (optional, not yet integrated)
- **pytest**: For Python tests

---

## Contributing Guidelines

### Before Submitting Code
1. Run all tests: `ctest && pytest`
2. Check code style
3. Add tests for new features
4. Update documentation

### Commit Messages
- Use present tense: "Add feature" not "Added feature"
- Be descriptive: Explain what and why
- Reference issues: "Fix #123: Memory leak in tensor copy"

---

## Useful Commands

```bash
# Build in debug mode
cmake .. -DCMAKE_BUILD_TYPE=Debug

# Build with verbose output
cmake --build . --verbose

# Run specific test
ctest -R test_tensor -V

# Generate compile_commands.json (for IDE integration)
cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Install to custom location
cmake --install . --prefix /custom/path

# Build Python wheel
python -m build

# Run Python tests with coverage
pytest --cov=pyflame_rt --cov-report=html
```

---

## Resources

### Documentation
- [docs/Developer_Guide.md](docs/Developer_Guide.md) - Complete developer guide
- [docs/API_Reference.md](docs/API_Reference.md) - API documentation
- [docs/Extending_PyFlameRT.md](docs/Extending_PyFlameRT.md) - Extension guide
- [docs/Examples_and_Tutorials.md](docs/Examples_and_Tutorials.md) - Examples

### External Resources
- [pybind11 documentation](https://pybind11.readthedocs.io/)
- [CMake documentation](https://cmake.org/documentation/)
- [GoogleTest documentation](https://google.github.io/googletest/)

---

## Contact and Support

- **GitHub Issues**: Report bugs and request features
- **Documentation**: See docs/ folder for detailed guides
- **Code Review**: All changes should be reviewed before merging

---

## License

[Add license information here]

---

## Acknowledgments

PyFlameRT is inspired by:
- NVIDIA TensorRT (high-performance inference)
- ONNX Runtime (cross-platform runtime)
- PyTorch (flexible tensor operations)
- Cerebras SDK (WSE-optimized inference)

---

*Last Updated: 2026-01-13*
*Version: 0.3.0 (Phase 3)*
