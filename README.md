# PyFlameRT

High-performance inference runtime for Cerebras WSE (Wafer-Scale Engine).

PyFlameRT provides an ONNX Runtime-compatible API for deploying deep learning models on Cerebras hardware, with a CPU reference backend for development and testing.

## Features

- **ONNX Runtime-compatible API** - Easy migration from existing ONNX Runtime code
- **C++ Core with Python Bindings** - High performance with convenient Python interface
- **CPU Reference Backend** - Development and testing without Cerebras hardware
- **Extensible Architecture** - Support for custom operators and backends

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/pyflame/pyflame-rt.git
cd pyflame-rt

# Build with CMake
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# Install Python package
pip install .
```

### Requirements

- CMake 3.18+
- C++17 compiler (GCC 8+, Clang 8+, MSVC 2019+)
- Python 3.9+
- NumPy 1.21+

## Quick Start

```python
import numpy as np
import pyflame_rt

# Create session options
options = pyflame_rt.SessionOptions()
options.device = "cpu"
options.num_threads = 4

# Load model
session = pyflame_rt.InferenceSession("model.pfm", options)

# Get input/output info
inputs = session.get_inputs()
outputs = session.get_outputs()
print(f"Input: {inputs[0].name}, shape: {inputs[0].shape}")
print(f"Output: {outputs[0].name}, shape: {outputs[0].shape}")

# Run inference
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
results = session.run(None, {"input": input_data})

print(f"Output shape: {results[0].shape}")
```

## Project Structure

```
PyFlameRT/
├── include/pyflame_rt/    # Public C++ headers
├── src/                   # C++ implementation
│   ├── backends/cpu/      # CPU reference backend
│   └── io/                # Model loading
├── bindings/              # pybind11 Python bindings
├── python/pyflame_rt/     # Python package
├── tests/                 # C++ and Python tests
└── docs/                  # Documentation
```

## Development

### Building

```bash
# Configure with tests
cmake -B build -DPYFLAME_RT_BUILD_TESTS=ON

# Build
cmake --build build --parallel

# Run C++ tests
cd build && ctest --output-on-failure

# Run Python tests
pytest tests/python/
```

### Supported Operators

PyFlameRT supports common neural network operators including:

- **Math**: Add, Sub, Mul, Div, MatMul, Gemm, Sqrt, Exp, Log, Pow
- **Activation**: Relu, Sigmoid, Tanh, Softmax, LeakyRelu, Gelu, SELU
- **Tensor**: Reshape, Transpose, Concat, Squeeze, Unsqueeze, Flatten
- **NN**: Conv, MaxPool, AveragePool, BatchNormalization, LayerNormalization, Dropout
- **Reduction**: ReduceSum, ReduceMean, ReduceMax, ReduceMin, ArgMax

## License

MIT License - see [LICENSE](LICENSE) for details.

## Related Projects

- [PyFlame](https://github.com/pyflame/pyflame) - Core tensor operations and IR graph
- [PyFlameVision](https://github.com/pyflame/pyflamevision) - Computer vision models and utilities
