# PyFlameRT API Reference

Complete API reference for PyFlameRT v0.7.0.

---

## Table of Contents

1. [Python API](#python-api)
   - [Module Overview](#module-overview)
   - [InferenceSession](#inferencesession)
   - [SessionOptions](#sessionoptions)
   - [RunOptions](#runoptions)
   - [CompileOptions](#compileoptions)
   - [Tensor](#tensor)
   - [TensorInfo](#tensorinfo)
   - [NodeArg](#nodearg)
   - [DType](#dtype)
   - [Functions](#functions)
   - [Model Import](#model-import)
   - [Quantization](#quantization)
   - [Pruning](#pruning)
   - [Distillation](#distillation)
   - [Custom Operators](#custom-operators)
   - [Partitioning](#partitioning)
   - [Exceptions](#exceptions)
2. [C++ API](#c-api)
   - [Namespace Structure](#namespace-structure)
   - [Core Types](#core-types)
   - [Tensor Class](#tensor-class)
   - [Node Class](#node-class)
   - [Graph Class](#graph-class)
   - [InferenceSession Class](#inferencesession-class-1)
   - [OperatorRegistry](#operatorregistry)
   - [Backend Interface](#backend-interface)
   - [Model Import (C++)](#model-import-c)
   - [Quantization (C++)](#quantization-c)
   - [Pruning (C++)](#pruning-c)
   - [Distillation (C++)](#distillation-c)
   - [Custom Operators (C++)](#custom-operators-c)
   - [Partitioning (C++)](#partitioning-c)
   - [Error Handling](#error-handling)
3. [Serving API](#serving-api)
   - [Python Client](#python-client)
   - [C++ Server](#c-server)
   - [Configuration Types](#configuration-types)
   - [Request/Response Types](#requestresponse-types)
   - [Model Registry](#model-registry)
   - [Metrics](#metrics)

---

## Python API

### Module Overview

```python
import pyflame_rt

# Version information
print(pyflame_rt.__version__)  # "0.1.0"

# Core classes
pyflame_rt.InferenceSession
pyflame_rt.SessionOptions
pyflame_rt.RunOptions
pyflame_rt.CompileOptions
pyflame_rt.Tensor
pyflame_rt.TensorInfo
pyflame_rt.NodeArg
pyflame_rt.DType

# Functions
pyflame_rt.from_numpy()
pyflame_rt.to_numpy()
pyflame_rt.get_available_providers()
pyflame_rt.get_device_count()

# Exceptions
pyflame_rt.PyFlameRTError
pyflame_rt.InvalidModelError
pyflame_rt.UnsupportedOperatorError
pyflame_rt.ValidationError
```

---

### InferenceSession

The main entry point for running inference with PyFlameRT models.

#### Constructor

```python
InferenceSession(
    model_path: str,
    options: SessionOptions = SessionOptions(),
    providers: List[str] = []
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | `str` | Required | Path to the `.pfm` model file |
| `options` | `SessionOptions` | `SessionOptions()` | Session configuration options |
| `providers` | `List[str]` | `[]` | Execution providers in priority order. Empty uses default |

**Raises:**
- `InvalidModelError`: If the model file is invalid or cannot be loaded
- `FileNotFoundError`: If the model file does not exist

**Example:**

```python
import pyflame_rt

# Basic usage
session = pyflame_rt.InferenceSession("model.pfm")

# With options
opts = pyflame_rt.SessionOptions()
opts.num_threads = 4
session = pyflame_rt.InferenceSession("model.pfm", opts)

# With specific provider
session = pyflame_rt.InferenceSession(
    "model.pfm",
    providers=["CPUExecutionProvider"]
)
```

#### Methods

##### run()

```python
run(
    output_names: Optional[List[str]],
    input_feed: Dict[str, numpy.ndarray],
    run_options: RunOptions = RunOptions()
) -> List[numpy.ndarray]
```

Execute the model with given inputs.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `output_names` | `Optional[List[str]]` | Names of outputs to return. `None` returns all outputs |
| `input_feed` | `Dict[str, numpy.ndarray]` | Dictionary mapping input names to numpy arrays |
| `run_options` | `RunOptions` | Optional run-time configuration |

**Returns:** `List[numpy.ndarray]` - Output tensors as numpy arrays

**Raises:**
- `ValidationError`: If inputs don't match model signature
- `RuntimeError`: If execution fails

**Example:**

```python
import numpy as np

# Run with all outputs
results = session.run(None, {"input": input_data})

# Run with specific outputs
results = session.run(["output1", "output2"], {"input": input_data})

# Run with options
opts = pyflame_rt.RunOptions()
opts.tag = "batch_1"
results = session.run(None, {"input": input_data}, opts)
```

##### get_inputs()

```python
get_inputs() -> List[NodeArg]
```

Get information about model inputs.

**Returns:** `List[NodeArg]` - List of input specifications

**Example:**

```python
inputs = session.get_inputs()
for inp in inputs:
    print(f"Name: {inp.name}")
    print(f"Shape: {inp.shape}")
    print(f"Type: {inp.type}")
```

##### get_outputs()

```python
get_outputs() -> List[NodeArg]
```

Get information about model outputs.

**Returns:** `List[NodeArg]` - List of output specifications

##### get_modelmeta()

```python
get_modelmeta() -> ModelMetadata
```

Get model metadata.

**Returns:** `ModelMetadata` - Model metadata object

**Example:**

```python
meta = session.get_modelmeta()
print(f"Producer: {meta.producer_name}")
print(f"Version: {meta.version}")
print(f"Domain: {meta.domain}")
print(f"Description: {meta.description}")
print(f"Custom metadata: {meta.custom_metadata}")
```

##### get_providers()

```python
get_providers() -> List[str]
```

Get list of execution providers used by this session.

**Returns:** `List[str]` - Provider names in execution order

##### get_profiling_start_time_ns()

```python
get_profiling_start_time_ns() -> int
```

Get profiling start time in nanoseconds (if profiling is enabled).

**Returns:** `int` - Timestamp in nanoseconds

##### end_profiling()

```python
end_profiling() -> str
```

End profiling and get the profile file path.

**Returns:** `str` - Path to the profiling output file

---

### SessionOptions

Configuration options for InferenceSession.

#### Constructor

```python
SessionOptions()
```

Creates a SessionOptions object with default values.

#### Properties

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `device` | `str` | `"cpu"` | Target device: `"cpu"`, `"wse"` |
| `num_threads` | `int` | `0` | Number of threads (0 = auto) |
| `enable_profiling` | `bool` | `False` | Enable execution profiling |
| `execution_mode` | `str` | `"sequential"` | Execution mode: `"sequential"`, `"parallel"` |
| `log_level` | `str` | `"warning"` | Log level: `"verbose"`, `"info"`, `"warning"`, `"error"`, `"fatal"` |
| `memory_limit` | `int` | `0` | Memory limit in bytes (0 = unlimited) |
| `optimization_level` | `int` | `2` | Optimization level (0-3) |

#### Methods

##### validate()

```python
validate() -> List[str]
```

Validate the options configuration.

**Returns:** `List[str]` - List of validation error messages (empty if valid)

**Example:**

```python
opts = pyflame_rt.SessionOptions()
opts.device = "invalid_device"
errors = opts.validate()
if errors:
    for error in errors:
        print(f"Error: {error}")
```

---

### RunOptions

Options for a single inference run.

#### Constructor

```python
RunOptions()
```

#### Properties

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `log_level` | `Optional[str]` | `None` | Override session log level for this run |
| `tag` | `Optional[str]` | `None` | Tag for profiling/debugging |
| `timeout_ms` | `Optional[int]` | `None` | Timeout in milliseconds |

**Example:**

```python
opts = pyflame_rt.RunOptions()
opts.tag = "inference_batch_42"
opts.timeout_ms = 5000  # 5 second timeout
results = session.run(None, inputs, opts)
```

---

### CompileOptions

Options for model compilation (used for model conversion).

#### Constructor

```python
CompileOptions()
```

#### Properties

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `cache_dir` | `Optional[str]` | `None` | Directory for compiled model cache |
| `dynamic_batch` | `bool` | `False` | Enable dynamic batch size |
| `optimization_level` | `int` | `2` | Compilation optimization level (0-3) |

---

### Tensor

Multi-dimensional array class for tensor data.

#### Creating Tensors

```python
# From numpy array
tensor = pyflame_rt.from_numpy(numpy_array)

# The tensor owns a copy of the data
```

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `shape` | `List[int]` | Tensor dimensions |
| `dtype` | `DType` | Data type |
| `ndim` | `int` | Number of dimensions |
| `num_elements` | `int` | Total number of elements |

#### Methods

##### numpy()

```python
numpy() -> numpy.ndarray
```

Convert tensor to numpy array (creates a copy).

**Returns:** `numpy.ndarray` - Copy of tensor data

##### clone()

```python
clone() -> Tensor
```

Create a deep copy of the tensor.

**Returns:** `Tensor` - New tensor with copied data

##### reshape()

```python
reshape(new_shape: List[int]) -> Tensor
```

Reshape tensor to new dimensions.

**Parameters:**
- `new_shape`: New shape (total elements must match)

**Returns:** `Tensor` - Reshaped tensor (may share data)

**Example:**

```python
import numpy as np
import pyflame_rt

arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
tensor = pyflame_rt.from_numpy(arr)

print(tensor.shape)        # [2, 3]
print(tensor.dtype)        # DType.Float32
print(tensor.ndim)         # 2
print(tensor.num_elements) # 6

reshaped = tensor.reshape([3, 2])
print(reshaped.shape)      # [3, 2]
```

---

### TensorInfo

Metadata describing a tensor's type and shape.

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `name` | `str` | Tensor name |
| `dtype` | `DType` | Data type |
| `shape` | `List[Optional[int]]` | Shape with optional dynamic dimensions (`None` = dynamic) |

---

### NodeArg

Describes an input or output of a model.

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `name` | `str` | Argument name |
| `shape` | `List[Optional[int]]` | Shape (None elements indicate dynamic dimensions) |
| `type` | `str` | Type string (e.g., "tensor(float)") |

**Example:**

```python
inputs = session.get_inputs()
for inp in inputs:
    print(f"Input: {inp.name}")
    print(f"  Shape: {inp.shape}")
    print(f"  Type: {inp.type}")

    # Handle dynamic dimensions
    concrete_shape = []
    for dim in inp.shape:
        if dim is None:
            concrete_shape.append(1)  # Use batch size 1
        else:
            concrete_shape.append(dim)
    print(f"  Concrete shape: {concrete_shape}")
```

---

### DType

Enumeration of supported data types.

#### Values

| Value | Integer | NumPy Equivalent |
|-------|---------|------------------|
| `DType.Float32` | 0 | `np.float32` |
| `DType.Float16` | 1 | `np.float16` |
| `DType.BFloat16` | 2 | N/A |
| `DType.Float64` | 3 | `np.float64` |
| `DType.Int64` | 4 | `np.int64` |
| `DType.Int32` | 5 | `np.int32` |
| `DType.Int16` | 6 | `np.int16` |
| `DType.Int8` | 7 | `np.int8` |
| `DType.UInt8` | 8 | `np.uint8` |
| `DType.Bool` | 9 | `np.bool_` |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `name` | `str` | Type name (e.g., "Float32") |
| `value` | `int` | Integer value |

**Example:**

```python
import pyflame_rt

dtype = pyflame_rt.DType.Float32
print(dtype.name)   # "Float32"
print(dtype.value)  # 0

# Check tensor dtype
tensor = pyflame_rt.from_numpy(np.zeros((2, 3), dtype=np.float32))
if tensor.dtype == pyflame_rt.DType.Float32:
    print("Tensor is float32")
```

---

### Functions

#### from_numpy()

```python
from_numpy(array: numpy.ndarray) -> Tensor
```

Create a Tensor from a numpy array.

**Parameters:**
- `array`: NumPy array to convert

**Returns:** `Tensor` - New tensor (owns a copy of the data)

**Supported dtypes:** float32, float64, float16, int64, int32, int16, int8, uint8, bool

#### to_numpy()

```python
to_numpy(tensor: Tensor) -> numpy.ndarray
```

Convert a Tensor to a numpy array.

**Parameters:**
- `tensor`: Tensor to convert

**Returns:** `numpy.ndarray` - Copy of tensor data

#### get_available_providers()

```python
get_available_providers() -> List[str]
```

Get list of available execution providers.

**Returns:** `List[str]` - Available provider names

**Example:**

```python
providers = pyflame_rt.get_available_providers()
print(providers)  # ["CPUExecutionProvider"]
```

#### get_device_count()

```python
get_device_count(device_type: str) -> int
```

Get number of available devices of a given type.

**Parameters:**
- `device_type`: Device type ("cpu", "wse")

**Returns:** `int` - Number of available devices

---

### Model Import

PyFlameRT provides importers for ONNX, PyTorch, and TorchScript models.

#### from_onnx()

```python
from_onnx(
    model_path: str,
    options: ImportOptions = ImportOptions()
) -> InferenceSession
```

Convenience function to import an ONNX model and create an inference session.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | `str` | Required | Path to ONNX model file (.onnx) |
| `options` | `ImportOptions` | `ImportOptions()` | Import configuration |

**Returns:** `InferenceSession` - Ready-to-use inference session

**Raises:**
- `InvalidModelError`: If the model file is invalid
- `UnsupportedOperatorError`: If model contains unsupported operators
- `FileNotFoundError`: If the model file does not exist

**Example:**

```python
import pyflame_rt
import numpy as np

# Quick import
session = pyflame_rt.from_onnx("resnet50.onnx")

# Run inference
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
results = session.run(None, {"input": input_data})
```

---

#### ImportOptions

Configuration for model import operations.

##### Constructor

```python
ImportOptions()
```

Creates an ImportOptions object with default values.

##### Properties

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `validate` | `bool` | `True` | Validate graph after import |
| `infer_shapes` | `bool` | `True` | Run shape inference |
| `optimize` | `bool` | `False` | Apply basic optimizations |
| `strict` | `bool` | `False` | Fail on warnings |

**Example:**

```python
options = pyflame_rt.ImportOptions()
options.validate = True
options.infer_shapes = True
options.optimize = True

session = pyflame_rt.from_onnx("model.onnx", options)
```

---

#### ImportResult

Result of a model import operation.

##### Properties

| Property | Type | Description |
|----------|------|-------------|
| `success` | `bool` | Whether import succeeded |
| `graph` | `Graph` | Imported graph (if successful) |
| `error` | `str` | Error message (if failed) |
| `stats` | `ImportStats` | Import statistics |

---

#### ImportStats

Statistics from a model import operation.

##### Properties

| Property | Type | Description |
|----------|------|-------------|
| `node_count` | `int` | Number of nodes imported |
| `initializer_count` | `int` | Number of initializers (weights) |
| `parameter_count` | `int` | Total number of parameters |
| `import_time_ms` | `float` | Time taken for import in milliseconds |
| `warnings` | `List[str]` | Warning messages generated during import |

**Example:**

```python
importer = pyflame_rt.import_module.ONNXImporter()
result = importer.import_model("model.onnx")

if result.success:
    stats = result.stats
    print(f"Imported {stats.node_count} nodes")
    print(f"Parameters: {stats.parameter_count:,}")
    print(f"Import time: {stats.import_time_ms:.2f}ms")
```

---

#### ONNXImporter

Importer for ONNX models.

##### Constructor

```python
pyflame_rt.import_module.ONNXImporter()
```

##### Methods

###### import_model()

```python
import_model(
    model_path: str,
    options: ImportOptions = ImportOptions()
) -> ImportResult
```

Import an ONNX model file.

**Parameters:**
- `model_path`: Path to ONNX model file
- `options`: Import options

**Returns:** `ImportResult` - Import result with graph and statistics

**Example:**

```python
importer = pyflame_rt.import_module.ONNXImporter()
result = importer.import_model("model.onnx")

if result.success:
    session = pyflame_rt.InferenceSession(result.graph)
```

###### supported_opsets()

```python
supported_opsets() -> List[int]
```

Get list of supported ONNX opset versions.

**Returns:** `List[int]` - Supported opset versions (9-21)

---

#### PyTorchImporter

Importer for PyTorch state dictionaries.

##### Constructor

```python
pyflame_rt.import_module.PyTorchImporter()
```

##### Methods

###### import_model()

```python
import_model(
    model_path: str,
    model_definer: Callable[[Graph, Dict[str, Tensor]], None],
    state_dict_key: Optional[str] = None,
    options: ImportOptions = ImportOptions()
) -> ImportResult
```

Import a PyTorch checkpoint file.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_path` | `str` | Path to PyTorch checkpoint (.pt, .pth) |
| `model_definer` | `Callable` | Function that defines the graph structure |
| `state_dict_key` | `Optional[str]` | Key to extract state dict from checkpoint |
| `options` | `ImportOptions` | Import options |

**Returns:** `ImportResult` - Import result with graph and statistics

**Example:**

```python
def define_model(graph, weights):
    # Define inputs
    graph.add_input("input", [None, 784], pyflame_rt.DType.Float32)

    # Add weights as initializers
    graph.add_initializer("fc.weight", weights["fc.weight"])
    graph.add_initializer("fc.bias", weights["fc.bias"])

    # Define operations
    graph.add_node("fc", "Gemm",
                   inputs=["input", "fc.weight", "fc.bias"],
                   outputs=["output"])

    # Define outputs
    graph.add_output("output", [None, 10], pyflame_rt.DType.Float32)

importer = pyflame_rt.import_module.PyTorchImporter()
result = importer.import_model("model.pth", define_model)
```

---

#### TorchScriptImporter

Importer for TorchScript models.

##### Constructor

```python
pyflame_rt.import_module.TorchScriptImporter()
```

##### Methods

###### import_model()

```python
import_model(
    model_path: str,
    options: ImportOptions = ImportOptions()
) -> ImportResult
```

Import a TorchScript model file.

**Parameters:**
- `model_path`: Path to TorchScript model (.pt)
- `options`: Import options

**Returns:** `ImportResult` - Import result with graph and statistics

**Example:**

```python
importer = pyflame_rt.import_module.TorchScriptImporter()
result = importer.import_model("traced_model.pt")

if result.success:
    session = pyflame_rt.InferenceSession(result.graph)

    # Note: TorchScript uses numbered names like "input.1"
    inputs = result.graph.inputs
    for inp in inputs:
        print(f"Input: {inp.name}")
```

---

### Quantization

PyFlameRT's quantization module provides tools for model compression and optimization.

#### QuantMode

Enumeration of quantization modes.

```python
from pyflame_rt.quantization import QuantMode

QuantMode.None_       # No quantization
QuantMode.FP16        # IEEE 754 half-precision
QuantMode.BFloat16    # Brain floating-point
QuantMode.DynamicInt8 # Dynamic INT8 quantization
QuantMode.StaticInt8  # Static INT8 with calibration
```

---

#### QuantGranularity

Quantization granularity options.

```python
from pyflame_rt.quantization import QuantGranularity

QuantGranularity.PerTensor  # One scale/zero-point per tensor
QuantGranularity.PerChannel # One scale/zero-point per output channel
```

---

#### CalibrationMethod

Calibration methods for static quantization.

```python
from pyflame_rt.quantization import CalibrationMethod

CalibrationMethod.MinMax     # Use min/max observed values
CalibrationMethod.Entropy    # KL-divergence minimization
CalibrationMethod.Percentile # Use percentile of values
```

---

#### QuantConfig

Configuration for quantization.

##### Constructor

```python
QuantConfig()
```

Creates a QuantConfig object with default values (no quantization).

##### Factory Methods

```python
# FP16 quantization
QuantConfig.fp16() -> QuantConfig

# BFloat16 quantization
QuantConfig.bfloat16() -> QuantConfig

# Dynamic INT8 quantization
QuantConfig.dynamic_int8() -> QuantConfig

# Static INT8 quantization with calibration
QuantConfig.static_int8(calibration_samples: int = 100) -> QuantConfig
```

##### Properties

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `mode` | `QuantMode` | `None_` | Quantization mode |
| `weight_dtype` | `DType` | `Int8` | Data type for quantized weights |
| `activation_dtype` | `DType` | `Int8` | Data type for quantized activations |
| `granularity` | `QuantGranularity` | `PerTensor` | Quantization granularity |
| `calibration_method` | `CalibrationMethod` | `MinMax` | Calibration method |
| `symmetric` | `bool` | `True` | Use symmetric quantization |
| `exclude_ops` | `List[str]` | `[]` | Operators to exclude from quantization |
| `calibration_samples` | `int` | `100` | Number of calibration samples |

##### Methods

###### is_valid()

```python
is_valid() -> bool
```

Check if configuration is valid.

**Returns:** `bool` - True if configuration is valid

###### validation_error()

```python
validation_error() -> str
```

Get validation error message.

**Returns:** `str` - Error message or empty string if valid

**Example:**

```python
from pyflame_rt.quantization import QuantConfig, QuantMode, QuantGranularity

# Create custom configuration
config = QuantConfig()
config.mode = QuantMode.StaticInt8
config.granularity = QuantGranularity.PerChannel
config.symmetric = False
config.exclude_ops = ["Softmax", "LayerNormalization"]

# Validate
if not config.is_valid():
    print(f"Invalid config: {config.validation_error()}")
```

---

#### QuantParams

Quantization parameters (scale and zero-point).

##### Factory Methods

```python
# Per-tensor parameters
QuantParams.per_tensor(
    scale: float,
    zero_point: int,
    dtype: DType = DType.Int8
) -> QuantParams

# Per-channel parameters
QuantParams.per_channel(
    scales: List[float],
    zero_points: List[int],
    channel_axis: int,
    dtype: DType = DType.Int8
) -> QuantParams

# Compute from min/max values
QuantParams.compute_from_minmax(
    min_val: float,
    max_val: float,
    dtype: DType,
    symmetric: bool = True
) -> QuantParams
```

##### Properties

| Property | Type | Description |
|----------|------|-------------|
| `scales` | `List[float]` | Scale factors |
| `zero_points` | `List[int]` | Zero-point offsets |
| `quantized_dtype` | `DType` | Target quantized dtype |
| `channel_axis` | `int` | Channel axis for per-channel (-1 for per-tensor) |

##### Methods

###### is_per_channel()

```python
is_per_channel() -> bool
```

Check if per-channel quantization.

###### num_channels()

```python
num_channels() -> int
```

Get number of channels.

**Example:**

```python
from pyflame_rt.quantization import QuantParams
import pyflame_rt

# Per-tensor parameters
params = QuantParams.per_tensor(0.01, 128, pyflame_rt.DType.Int8)

# Per-channel parameters
scales = [0.01, 0.02, 0.015]
zero_points = [128, 130, 125]
params = QuantParams.per_channel(scales, zero_points, 0, pyflame_rt.DType.Int8)

# Compute from data range
params = QuantParams.compute_from_minmax(-1.0, 1.0, pyflame_rt.DType.Int8, symmetric=True)
```

---

#### GraphQuantInfo

Graph-level quantization information.

##### Properties

| Property | Type | Description |
|----------|------|-------------|
| `tensor_params` | `Dict[str, QuantParams]` | Params for each tensor |
| `weights_quantized` | `bool` | Whether weights are quantized |
| `activations_quantized` | `bool` | Whether activations are quantized |

##### Methods

###### has_params()

```python
has_params(tensor_name: str) -> bool
```

Check if parameters exist for a tensor.

###### get_params()

```python
get_params(tensor_name: str) -> QuantParams
```

Get parameters for a tensor.

**Raises:** `KeyError` if tensor not found

---

#### QuantizationResult

Result of a quantization operation.

##### Properties

| Property | Type | Description |
|----------|------|-------------|
| `success` | `bool` | Whether quantization succeeded |
| `quantized_graph` | `Graph` | Quantized graph (if successful) |
| `quant_info` | `GraphQuantInfo` | Quantization parameters |
| `error_message` | `str` | Error message (if failed) |
| `stats` | `QuantizationStats` | Quantization statistics |

---

#### QuantizationStats

Statistics from quantization.

##### Properties

| Property | Type | Description |
|----------|------|-------------|
| `nodes_quantized` | `int` | Number of nodes quantized |
| `nodes_skipped` | `int` | Number of nodes skipped |
| `weights_quantized` | `int` | Number of weight tensors quantized |
| `original_size_bytes` | `int` | Original model size in bytes |
| `quantized_size_bytes` | `int` | Quantized model size in bytes |

##### Methods

###### compression_ratio()

```python
compression_ratio() -> float
```

Get compression ratio.

**Returns:** `float` - Ratio of original to quantized size

###### original_size_mb()

```python
original_size_mb() -> float
```

Get original size in megabytes.

###### quantized_size_mb()

```python
quantized_size_mb() -> float
```

Get quantized size in megabytes.

---

#### Quantizer

Main class for graph quantization.

##### Constructor

```python
Quantizer(config: QuantConfig)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `QuantConfig` | Quantization configuration |

##### Methods

###### convert_to_fp16()

```python
convert_to_fp16(graph: Graph) -> QuantizationResult
```

Convert graph to FP16.

**Parameters:**
- `graph`: Input graph

**Returns:** `QuantizationResult` - Result with FP16 graph

###### convert_to_bfloat16()

```python
convert_to_bfloat16(graph: Graph) -> QuantizationResult
```

Convert graph to BFloat16.

###### quantize_dynamic()

```python
quantize_dynamic(graph: Graph) -> QuantizationResult
```

Apply dynamic INT8 quantization.

###### quantize()

```python
quantize(graph: Graph, quant_info: GraphQuantInfo) -> QuantizationResult
```

Apply static quantization with pre-computed parameters.

**Parameters:**
- `graph`: Input graph
- `quant_info`: Pre-computed quantization parameters

**Returns:** `QuantizationResult` - Result with quantized graph

###### quantize_with_calibration()

```python
quantize_with_calibration(
    graph: Graph,
    data_provider: Callable[[], Dict[str, ndarray]],
    num_batches: int
) -> QuantizationResult
```

Quantize with inline calibration.

**Example:**

```python
from pyflame_rt.quantization import Quantizer, QuantConfig

# FP16 quantization
config = QuantConfig.fp16()
quantizer = Quantizer(config)
result = quantizer.convert_to_fp16(graph)

# Dynamic INT8
config = QuantConfig.dynamic_int8()
quantizer = Quantizer(config)
result = quantizer.quantize_dynamic(graph)

# Static INT8 with calibration
def get_data():
    return {"input": np.random.randn(1, 3, 224, 224).astype(np.float32)}

config = QuantConfig.static_int8()
quantizer = Quantizer(config)
result = quantizer.quantize_with_calibration(graph, get_data, 100)
```

---

#### Calibrator

Calibrator for static quantization.

##### Constructor

```python
Calibrator(graph: Graph, config: QuantConfig)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `graph` | `Graph` | Graph to calibrate |
| `config` | `QuantConfig` | Quantization configuration |

##### Methods

###### observe()

```python
observe(input_feed: Dict[str, ndarray]) -> None
```

Feed calibration data to collect statistics.

**Parameters:**
- `input_feed`: Dictionary mapping input names to numpy arrays

###### compute_quant_params()

```python
compute_quant_params() -> GraphQuantInfo
```

Compute quantization parameters from collected statistics.

**Returns:** `GraphQuantInfo` - Computed parameters for all tensors

###### calibrate()

```python
calibrate(
    data_provider: Callable[[], Dict[str, ndarray]],
    num_batches: int
) -> None
```

Run full calibration.

**Parameters:**
- `data_provider`: Callable that returns calibration data
- `num_batches`: Number of batches to process

###### get_stats()

```python
get_stats(tensor_name: str) -> CalibrationStats
```

Get calibration statistics for a tensor.

**Example:**

```python
from pyflame_rt.quantization import Calibrator, QuantConfig

# Create calibrator
config = QuantConfig.static_int8()
calibrator = Calibrator(graph, config)

# Feed calibration data
for i in range(100):
    data = {"input": load_calibration_image(i)}
    calibrator.observe(data)

# Compute parameters
quant_info = calibrator.compute_quant_params()

# Use parameters for quantization
quantizer = Quantizer(config)
result = quantizer.quantize(graph, quant_info)
```

---

#### QuantizationReport

Report from session quantization (returned by `InferenceSession.quantization_report()`).

##### Properties

| Property | Type | Description |
|----------|------|-------------|
| `mode` | `QuantMode` | Quantization mode used |
| `nodes_quantized` | `int` | Number of nodes quantized |
| `nodes_total` | `int` | Total number of nodes |
| `compression_ratio` | `float` | Size compression ratio |
| `original_size_mb` | `float` | Original model size (MB) |
| `quantized_size_mb` | `float` | Quantized model size (MB) |
| `weights_quantized` | `bool` | Whether weights were quantized |
| `activations_quantized` | `bool` | Whether activations were quantized |

---

### Pruning

Weight pruning module for model compression.

```python
from pyflame_rt import pruning
```

#### PruningGranularity

```python
class PruningGranularity(Enum):
    UNSTRUCTURED = 0    # Individual weight pruning
    STRUCTURED = 1      # Channel/filter pruning
    BLOCK = 2          # Block-sparse pruning
    NM = 3             # N:M sparsity pattern
```

#### PruningCriterion

```python
class PruningCriterion(Enum):
    MAGNITUDE = 0      # Prune smallest magnitude weights
    MOVEMENT = 1       # Movement pruning
    RANDOM = 2         # Random pruning
    TAYLOR = 3         # Taylor expansion based
    LAMP = 4           # Layer-Adaptive Magnitude Pruning
```

#### PruningSchedule

```python
class PruningSchedule(Enum):
    ONE_SHOT = 0       # Prune all at once
    ITERATIVE = 1      # Gradually increase sparsity
    CUBIC = 2          # Cubic sparsity schedule
    POLYNOMIAL = 3     # Polynomial schedule
```

#### PruningConfig

Configuration for weight pruning.

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `target_sparsity` | `float` | 0.5 | Target sparsity (0.0-1.0) |
| `granularity` | `PruningGranularity` | UNSTRUCTURED | Pruning granularity |
| `criterion` | `PruningCriterion` | MAGNITUDE | Weight importance criterion |
| `schedule` | `PruningSchedule` | ONE_SHOT | Pruning schedule |
| `n_value` | `int` | 2 | N value for N:M sparsity |
| `m_value` | `int` | 4 | M value for N:M sparsity |
| `start_step` | `int` | 0 | Step to start pruning |
| `end_step` | `int` | 1000 | Step to end pruning |

**Static Methods:**

| Method | Description |
|--------|-------------|
| `magnitude_pruning(sparsity)` | Create magnitude-based config |
| `structured_pruning(sparsity)` | Create structured pruning config |
| `nm_sparsity(n, m)` | Create N:M sparsity config |

#### WeightPruner

Main class for pruning weights.

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `prune(graph)` | `graph: Graph` | `Graph` | Prune weights in graph |
| `compute_masks(graph)` | `graph: Graph` | `Dict[str, PruningMask]` | Compute pruning masks |
| `apply_masks(graph, masks)` | `graph: Graph, masks: Dict` | `Graph` | Apply masks to graph |
| `get_sparsity_at_step(step)` | `step: int` | `float` | Get target sparsity at step |
| `get_stats()` | - | `PruningStats` | Get pruning statistics |

#### SparseTensor

Sparse tensor representation.

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `from_dense(tensor, format)` | `tensor: Tensor, format: SparseFormat` | `SparseTensor` | Create from dense |
| `to_dense()` | - | `Tensor` | Convert to dense |
| `to_coo()` | - | `SparseTensor` | Convert to COO format |
| `to_csr()` | - | `SparseTensor` | Convert to CSR format |
| `nnz()` | - | `int` | Number of non-zeros |
| `sparsity()` | - | `float` | Sparsity ratio |
| `memory_bytes()` | - | `int` | Memory usage |
| `compression_ratio()` | - | `float` | Compression vs dense |

---

### Distillation

Knowledge distillation module for training smaller models.

```python
from pyflame_rt import distillation
```

#### DistillationLoss

```python
class DistillationLoss(Enum):
    KL_DIVERGENCE = 0  # KL divergence for soft targets
    MSE = 1            # Mean squared error
    COSINE = 2         # Cosine similarity
    ATTENTION = 3      # Attention transfer
    HINT = 4           # Hint-based feature distillation
```

#### DistillationConfig

Configuration for knowledge distillation.

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `temperature` | `float` | 4.0 | Softmax temperature |
| `alpha` | `float` | 0.7 | Weight for distillation loss |
| `loss_type` | `DistillationLoss` | KL_DIVERGENCE | Loss function |
| `use_hard_labels` | `bool` | True | Use hard labels too |
| `feature_layers` | `List[str]` | [] | Layers for feature distillation |
| `normalize_features` | `bool` | False | Normalize features |

**Static Methods:**

| Method | Description |
|--------|-------------|
| `soft_label(temperature)` | Create soft label config |
| `feature_distillation(layers)` | Create feature distillation config |
| `attention_transfer(layers)` | Create attention transfer config |

#### StudentConfig

Configuration for student model architecture.

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `hidden_dim_ratio` | `float` | 1.0 | Ratio to teacher hidden dim |
| `num_layers_ratio` | `float` | 1.0 | Ratio to teacher layers |
| `num_heads_ratio` | `float` | 1.0 | Ratio to teacher heads |

**Static Methods:**

| Method | Description |
|--------|-------------|
| `half_size()` | Create half-size student config |
| `quarter_size()` | Create quarter-size student config |

#### TrainingConfig

Configuration for distillation training.

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `learning_rate` | `float` | 1e-4 | Initial learning rate |
| `batch_size` | `int` | 32 | Training batch size |
| `num_epochs` | `int` | 10 | Number of epochs |
| `warmup_steps` | `int` | 0 | Warmup steps |
| `weight_decay` | `float` | 0.0 | Weight decay |
| `gradient_clip` | `float` | 1.0 | Gradient clipping |
| `early_stopping_patience` | `int` | 0 | Early stopping patience |

#### DistillationTrainer

Trainer for knowledge distillation.

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `set_training_config(config)` | `config: TrainingConfig` | - | Set training config |
| `set_dataset(dataset)` | `dataset: DistillationDataset` | - | Set training dataset |
| `train(callback)` | `callback: Optional[Callable]` | `DistillationResult` | Run training |
| `get_student()` | - | `Graph` | Get current student |

#### InMemoryDataset

In-memory dataset for distillation.

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `add_sample(inputs, labels)` | `inputs: Dict, labels: Dict` | - | Add training sample |
| `size()` | - | `int` | Get dataset size |
| `clear()` | - | - | Clear all samples |
| `shuffle()` | - | - | Shuffle dataset |
| `get_batch(start, batch_size)` | `start: int, batch_size: int` | `List` | Get batch |

---

### Custom Operators

Custom operator registration module.

```python
from pyflame_rt import custom
```

#### BackendType

```python
class BackendType(Enum):
    CPU = 0    # CPU execution
    WSE = 1    # Cerebras WSE
    CUDA = 2   # NVIDIA CUDA
    ALL = 3    # All backends
```

#### CustomOpBuilder

Fluent builder for custom operators.

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `domain(name)` | `name: str` | `CustomOpBuilder` | Set operator domain |
| `version(v)` | `v: int` | `CustomOpBuilder` | Set operator version |
| `doc(text)` | `text: str` | `CustomOpBuilder` | Set documentation |
| `input(name, dtype, optional)` | `name: str, dtype: DType, optional: bool` | `CustomOpBuilder` | Add input |
| `output(name, dtype)` | `name: str, dtype: DType` | `CustomOpBuilder` | Add output |
| `attr_int(name, required)` | `name: str, required: bool` | `CustomOpBuilder` | Add int attribute |
| `attr_float(name, required)` | `name: str, required: bool` | `CustomOpBuilder` | Add float attribute |
| `attr_string(name, required)` | `name: str, required: bool` | `CustomOpBuilder` | Add string attribute |
| `kernel(fn, backend)` | `fn: Callable, backend: BackendType` | `CustomOpBuilder` | Set kernel function |
| `shape_inference(fn)` | `fn: Callable` | `CustomOpBuilder` | Set shape inference |
| `gradient(fn)` | `fn: Callable` | `CustomOpBuilder` | Set gradient function |
| `build()` | - | `CustomOp` | Build and register |

#### CustomOpRegistry

Registry for custom operators.

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `instance()` | - | `CustomOpRegistry` | Get singleton |
| `register_op(schema)` | `schema: OpSchema` | `CustomOp` | Register operator |
| `get(name)` | `name: str` | `Optional[CustomOp]` | Get operator |
| `has(name)` | `name: str` | `bool` | Check if exists |
| `list()` | - | `List[str]` | List all operators |
| `unregister(name)` | `name: str` | - | Unregister operator |
| `clear()` | - | - | Clear all operators |
| `size()` | - | `int` | Number of operators |

#### CustomOp

Custom operator instance.

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `name()` | - | `str` | Get operator name |
| `domain()` | - | `str` | Get operator domain |
| `full_name()` | - | `str` | Get fully qualified name |
| `execute(inputs, attrs)` | `inputs: List[Tensor], attrs: Dict` | `List[Tensor]` | Execute operator |
| `has_gradient()` | - | `bool` | Check if has gradient |
| `gradient(inputs, grad_outputs)` | `inputs: List, grad_outputs: List` | `List[Tensor]` | Compute gradients |
| `supports_backend(backend)` | `backend: BackendType` | `bool` | Check backend support |

---

### Partitioning

Graph partitioning module for multi-device execution.

```python
from pyflame_rt import partition
```

#### DeviceType

```python
class DeviceType(Enum):
    CPU = 0         # CPU device
    WSE = 1         # Cerebras WSE
    GPU = 2         # GPU device
    DISTRIBUTED = 3 # Distributed multi-node
```

#### PartitionStrategy

```python
class PartitionStrategy(Enum):
    MANUAL = 0          # User-specified
    DATA_PARALLEL = 1   # Replicate model, split data
    MODEL_PARALLEL = 2  # Split model across devices
    PIPELINE_PARALLEL = 3 # Pipeline stages
    HYBRID = 4          # Combination
    AUTOMATIC = 5       # Cost-model based
```

#### DeviceSpec

Device specification.

| Property | Type | Description |
|----------|------|-------------|
| `type` | `DeviceType` | Device type |
| `device_id` | `int` | Device ID |
| `memory_bytes` | `int` | Available memory |
| `compute_flops` | `float` | Compute capacity |
| `name` | `str` | Device name |

#### WSEChipConfig

WSE multi-chip configuration.

| Property | Type | Description |
|----------|------|-------------|
| `num_chips` | `int` | Number of chips |
| `topology` | `List[int]` | Chip topology (e.g., [2, 2]) |
| `inter_chip_bandwidth` | `float` | Bandwidth (bytes/sec) |
| `inter_chip_latency` | `float` | Latency (nanoseconds) |
| `chip_memory_bytes` | `int` | Memory per chip |

#### PartitionConfig

Configuration for graph partitioning.

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `devices` | `List[DeviceSpec]` | [] | Target devices |
| `strategy` | `PartitionStrategy` | AUTOMATIC | Partitioning strategy |
| `balance_compute` | `bool` | True | Balance compute load |
| `minimize_communication` | `bool` | True | Minimize communication |
| `max_pipeline_stages` | `int` | 0 | Max pipeline stages |
| `micro_batch_size` | `int` | 1 | Micro-batch size |

#### GraphPartitioner

Graph partitioner for multi-device execution.

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `partition(graph)` | `graph: Graph` | `PartitionPlan` | Partition graph |
| `partition_data_parallel(graph, n)` | `graph: Graph, n: int` | `PartitionPlan` | Data parallel |
| `partition_model_parallel(graph, n)` | `graph: Graph, n: int` | `PartitionPlan` | Model parallel |
| `partition_pipeline(graph, n)` | `graph: Graph, n: int` | `PartitionPlan` | Pipeline parallel |
| `analyze(graph)` | `graph: Graph` | `AnalysisResult` | Analyze graph |
| `set_cost_model(model)` | `model: CostModel` | - | Set cost model |

**Static Methods:**

| Method | Description |
|--------|-------------|
| `validate(plan)` | Validate partition plan |
| `get_stats(plan)` | Get partition statistics |
| `estimate_execution_time(plan)` | Estimate execution time |

#### PartitionPlan

Complete partition plan.

| Property | Type | Description |
|----------|------|-------------|
| `partitions` | `List[GraphPartition]` | List of partitions |
| `communications` | `List[CommOp]` | Communication operations |
| `total_latency_us` | `float` | Total estimated latency |
| `total_comm_bytes` | `int` | Total communication bytes |
| `load_imbalance` | `float` | Load imbalance metric |

#### PartitionedExecutor

Executor for partitioned graphs.

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `execute(inputs)` | `inputs: Dict[str, Tensor]` | `Dict[str, Tensor]` | Execute graph |
| `execute_async(inputs)` | `inputs: Dict[str, Tensor]` | `Future` | Async execution |

#### CostModel

Cost model for partitioning decisions.

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `estimate_compute_cost(node, device)` | `node: Node, device: DeviceSpec` | `float` | Compute cost (us) |
| `estimate_memory_cost(node)` | `node: Node` | `int` | Memory cost (bytes) |
| `estimate_comm_cost(bytes, src, dst)` | `bytes: int, src: DeviceSpec, dst: DeviceSpec` | `float` | Comm cost (us) |
| `set_inter_device_bandwidth(bw)` | `bw: float` | - | Set bandwidth |
| `set_inter_device_latency(lat)` | `lat: float` | - | Set latency |

#### Convenience Functions

| Function | Parameters | Returns | Description |
|----------|------------|---------|-------------|
| `auto_partition(graph, num_devices)` | `graph: Graph, num_devices: int` | `PartitionPlan` | Auto partition |
| `partition_for_multi_chip(graph, config)` | `graph: Graph, config: WSEChipConfig` | `PartitionPlan` | Partition for WSE |

#### WSE Submodule

```python
from pyflame_rt.partition import wse
```

| Function | Parameters | Returns | Description |
|----------|------------|---------|-------------|
| `partition_for_wse(graph, config)` | `graph: Graph, config: WSEChipConfig` | `PartitionPlan` | Partition for WSE |
| `insert_wse_communication(graph, plan)` | `graph: Graph, plan: PartitionPlan` | `Graph` | Insert WSE comm ops |
| `optimize_for_wse_dataflow(graph)` | `graph: Graph` | `Graph` | Optimize for WSE |

---

### Exceptions

All PyFlameRT exceptions inherit from `PyFlameRTError`.

#### PyFlameRTError

```python
class PyFlameRTError(Exception):
    """Base exception for all PyFlameRT errors."""
    pass
```

#### InvalidModelError

```python
class InvalidModelError(PyFlameRTError):
    """Raised when a model file is invalid or corrupted."""
    pass
```

**Common causes:**
- Invalid file format
- Corrupted model data
- Version mismatch

#### UnsupportedOperatorError

```python
class UnsupportedOperatorError(PyFlameRTError):
    """Raised when a model contains unsupported operators."""
    pass
```

**Common causes:**
- Model uses operators not implemented in PyFlameRT
- Operator attributes not supported

#### ValidationError

```python
class ValidationError(PyFlameRTError):
    """Raised when input validation fails."""
    pass
```

**Common causes:**
- Input shape mismatch
- Input dtype mismatch
- Missing required inputs

**Example:**

```python
import pyflame_rt

try:
    session = pyflame_rt.InferenceSession("model.pfm")
    results = session.run(None, {"input": data})
except pyflame_rt.InvalidModelError as e:
    print(f"Invalid model: {e}")
except pyflame_rt.ValidationError as e:
    print(f"Validation error: {e}")
except pyflame_rt.PyFlameRTError as e:
    print(f"PyFlameRT error: {e}")
```

---

## C++ API

### Namespace Structure

```cpp
namespace pyflame_rt {
    // Core types
    enum class DType;
    using Shape = std::vector<std::optional<int64_t>>;
    struct TensorInfo;
    struct NodeArg;
    struct ModelMetadata;

    // Core classes
    class Tensor;
    class Node;
    class Graph;
    class InferenceSession;

    // Options
    struct SessionOptions;
    struct RunOptions;
    struct CompileOptions;

    // Registry
    class OperatorRegistry;

    // Backend interface
    class Backend;

    // Exceptions
    class PyFlameRTError;
    class InvalidModelError;
    class UnsupportedOperatorError;
    class ValidationError;
}
```

---

### Core Types

#### DType

```cpp
enum class DType : uint8_t {
    Float32 = 0,
    Float16 = 1,
    BFloat16 = 2,
    Float64 = 3,
    Int64 = 4,
    Int32 = 5,
    Int16 = 6,
    Int8 = 7,
    UInt8 = 8,
    Bool = 9
};

// Helper functions
size_t dtype_size(DType dtype);           // Size in bytes
std::string dtype_to_string(DType dtype); // String representation
DType string_to_dtype(const std::string& str);
```

#### Shape

```cpp
using Shape = std::vector<std::optional<int64_t>>;

// Examples:
Shape static_shape = {1, 3, 224, 224};           // All static
Shape dynamic_batch = {std::nullopt, 3, 224, 224}; // Dynamic batch
```

#### TensorInfo

```cpp
struct TensorInfo {
    std::string name;
    DType dtype = DType::Float32;
    Shape shape;
};
```

#### NodeArg

```cpp
struct NodeArg {
    std::string name;
    DType dtype = DType::Float32;
    Shape shape;

    std::string type_str() const;  // e.g., "tensor(float)"
};
```

#### ModelMetadata

```cpp
struct ModelMetadata {
    std::string producer_name;
    std::string producer_version;
    std::string domain;
    int64_t model_version = 0;
    std::string doc_string;
    std::unordered_map<std::string, std::string> custom_metadata;
};
```

---

### Tensor Class

```cpp
class Tensor {
public:
    // Constructors
    Tensor();
    Tensor(const std::vector<int64_t>& shape, DType dtype);
    Tensor(void* data, const std::vector<int64_t>& shape, DType dtype);

    // Copy/Move
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;

    // Destructor
    ~Tensor();

    // Data access
    void* data();
    const void* data() const;

    template<typename T>
    T* data_ptr();

    template<typename T>
    const T* data_ptr() const;

    // Properties
    const std::vector<int64_t>& shape() const;
    DType dtype() const;
    size_t ndim() const;
    int64_t num_elements() const;
    size_t size_bytes() const;
    bool empty() const;
    bool owns_data() const;

    // Operations
    Tensor clone() const;
    Tensor view() const;
    Tensor reshape(const std::vector<int64_t>& new_shape) const;
};
```

#### Constructor Details

```cpp
// Create uninitialized tensor with shape and dtype
Tensor tensor({1, 3, 224, 224}, DType::Float32);

// Create tensor from external data (non-owning view)
float* external_data = ...;
Tensor view(external_data, {batch, channels, height, width}, DType::Float32);

// Create owning copy from external data
Tensor owned = view.clone();
```

#### Data Access

```cpp
// Generic pointer
void* ptr = tensor.data();

// Typed pointer (no runtime check)
float* fptr = tensor.data_ptr<float>();
int64_t* iptr = tensor.data_ptr<int64_t>();

// Example: Fill tensor with value
Tensor tensor({2, 3}, DType::Float32);
float* data = tensor.data_ptr<float>();
for (int64_t i = 0; i < tensor.num_elements(); ++i) {
    data[i] = 1.0f;
}
```

---

### Node Class

Represents a single operation in the computation graph.

```cpp
class Node {
public:
    // Constructors
    Node(const std::string& op_type,
         const std::string& name,
         const std::vector<std::string>& inputs,
         const std::vector<std::string>& outputs);

    // Properties
    const std::string& op_type() const;
    const std::string& name() const;
    const std::vector<std::string>& inputs() const;
    const std::vector<std::string>& outputs() const;

    // Attributes
    void set_attribute(const std::string& name, int64_t value);
    void set_attribute(const std::string& name, double value);
    void set_attribute(const std::string& name, const std::string& value);
    void set_attribute(const std::string& name, const std::vector<int64_t>& value);
    void set_attribute(const std::string& name, const std::vector<double>& value);

    template<typename T>
    T get_attribute(const std::string& name) const;

    template<typename T>
    T get_attribute(const std::string& name, const T& default_value) const;

    bool has_attribute(const std::string& name) const;
    std::vector<std::string> attribute_names() const;
};
```

#### Example Usage

```cpp
// Create a Conv node
auto conv = std::make_shared<Node>(
    "Conv",
    "conv1",
    {"input", "weight", "bias"},
    {"conv1_output"}
);

// Set attributes
conv->set_attribute("kernel_shape", std::vector<int64_t>{3, 3});
conv->set_attribute("strides", std::vector<int64_t>{1, 1});
conv->set_attribute("pads", std::vector<int64_t>{1, 1, 1, 1});
conv->set_attribute("group", int64_t{1});

// Read attributes
auto kernel = conv->get_attribute<std::vector<int64_t>>("kernel_shape");
auto group = conv->get_attribute<int64_t>("group", 1);  // With default
```

---

### Graph Class

Container for the computation graph.

```cpp
class Graph {
public:
    // Construction
    Graph();
    explicit Graph(const std::string& name);

    // Graph building
    void add_node(std::shared_ptr<Node> node);
    void add_input(const TensorInfo& info);
    void add_output(const TensorInfo& info);
    void add_initializer(const std::string& name, Tensor tensor);

    // Properties
    const std::string& name() const;
    const std::vector<std::shared_ptr<Node>>& nodes() const;
    const std::vector<TensorInfo>& inputs() const;
    const std::vector<TensorInfo>& outputs() const;

    // Initializers
    bool has_initializer(const std::string& name) const;
    const Tensor& get_initializer(const std::string& name) const;
    const std::unordered_map<std::string, Tensor>& initializers() const;

    // Analysis
    std::vector<std::shared_ptr<Node>> topological_sort() const;
    std::vector<std::string> validate() const;

    // Metadata
    ModelMetadata& metadata();
    const ModelMetadata& metadata() const;
};
```

#### Example: Building a Graph

```cpp
Graph graph("simple_model");

// Define inputs
graph.add_input(TensorInfo{"input", DType::Float32, {1, 3, 224, 224}});

// Add initializers (weights)
Tensor weight({64, 3, 7, 7}, DType::Float32);
// ... initialize weight data ...
graph.add_initializer("conv1_weight", std::move(weight));

// Add nodes
auto conv = std::make_shared<Node>("Conv", "conv1",
    std::vector<std::string>{"input", "conv1_weight"},
    std::vector<std::string>{"conv1_out"});
conv->set_attribute("kernel_shape", std::vector<int64_t>{7, 7});
graph.add_node(conv);

auto relu = std::make_shared<Node>("Relu", "relu1",
    std::vector<std::string>{"conv1_out"},
    std::vector<std::string>{"relu1_out"});
graph.add_node(relu);

// Define outputs
graph.add_output(TensorInfo{"relu1_out", DType::Float32, {1, 64, 218, 218}});

// Validate
auto errors = graph.validate();
if (!errors.empty()) {
    for (const auto& err : errors) {
        std::cerr << "Validation error: " << err << "\n";
    }
}
```

---

### InferenceSession Class

```cpp
class InferenceSession {
public:
    // Constructors
    explicit InferenceSession(const std::string& model_path,
                              SessionOptions options = {},
                              std::vector<std::string> providers = {});

    explicit InferenceSession(std::shared_ptr<Graph> graph,
                              SessionOptions options = {},
                              std::vector<std::string> providers = {});

    // Destructor
    ~InferenceSession();

    // Inference
    std::vector<Tensor> run(
        const std::vector<std::string>& output_names,
        const std::unordered_map<std::string, Tensor>& input_feed,
        const RunOptions& run_options = {});

    // Model info
    std::vector<NodeArg> get_inputs() const;
    std::vector<NodeArg> get_outputs() const;
    ModelMetadata get_modelmeta() const;
    std::vector<std::string> get_providers() const;

    // Profiling
    int64_t get_profiling_start_time_ns() const;
    std::string end_profiling();
};
```

#### Example Usage

```cpp
#include <pyflame_rt/session.hpp>

// Create session
SessionOptions opts;
opts.device = "cpu";
opts.num_threads = 4;

InferenceSession session("model.pfm", opts);

// Get input info
auto inputs = session.get_inputs();
std::cout << "Input: " << inputs[0].name
          << ", shape: " << shape_to_string(inputs[0].shape) << "\n";

// Prepare input
Tensor input({1, 3, 224, 224}, DType::Float32);
// ... fill input data ...

// Run inference
auto outputs = session.run({}, {{"input", input}});

// Process output
const float* output_data = outputs[0].data_ptr<float>();
```

---

### OperatorRegistry

Singleton registry for operator implementations.

```cpp
class OperatorRegistry {
public:
    // Singleton access
    static OperatorRegistry& instance();

    // Registration
    using OpFunction = std::function<Tensor(
        const std::vector<Tensor>& inputs,
        const std::unordered_map<std::string, std::any>& attributes)>;

    void register_op(const std::string& op_type, OpFunction func);

    // Execution
    bool has_op(const std::string& op_type) const;

    Tensor execute(
        const std::string& op_type,
        const std::vector<Tensor>& inputs,
        const std::unordered_map<std::string, std::any>& attributes) const;

    // Inspection
    std::vector<std::string> registered_ops() const;
};
```

#### Registering Custom Operators

```cpp
#include <pyflame_rt/registry.hpp>

// Define operator function
Tensor my_custom_op(
    const std::vector<Tensor>& inputs,
    const std::unordered_map<std::string, std::any>& attrs
) {
    const Tensor& input = inputs[0];
    Tensor output(input.shape(), input.dtype());

    // Implement operation...
    const float* in_data = input.data_ptr<float>();
    float* out_data = output.data_ptr<float>();

    for (int64_t i = 0; i < input.num_elements(); ++i) {
        out_data[i] = /* custom computation */;
    }

    return output;
}

// Register at static initialization
struct MyOpRegistrar {
    MyOpRegistrar() {
        OperatorRegistry::instance().register_op("MyCustomOp", my_custom_op);
    }
};
static MyOpRegistrar registrar;
```

---

### Backend Interface

Abstract interface for execution backends.

```cpp
class Backend {
public:
    virtual ~Backend() = default;

    // Backend info
    virtual std::string name() const = 0;
    virtual std::string device_type() const = 0;

    // Initialization
    virtual bool initialize(const SessionOptions& options) = 0;
    virtual void shutdown() = 0;

    // Execution
    virtual std::vector<Tensor> execute(
        const Graph& graph,
        const std::unordered_map<std::string, Tensor>& inputs) = 0;

    // Memory management
    virtual Tensor allocate(const std::vector<int64_t>& shape, DType dtype) = 0;
    virtual void deallocate(Tensor& tensor) = 0;

    // Capabilities
    virtual bool supports_op(const std::string& op_type) const = 0;
    virtual std::vector<std::string> supported_ops() const = 0;
};
```

---

### Model Import (C++)

C++ API for importing models from various formats.

#### ImportOptions

```cpp
struct ImportOptions {
    bool validate = true;       // Validate graph after import
    bool infer_shapes = true;   // Run shape inference
    bool optimize = false;      // Apply basic optimizations
    bool strict = false;        // Fail on warnings
};
```

#### ImportStats

```cpp
struct ImportStats {
    size_t node_count = 0;           // Number of nodes imported
    size_t initializer_count = 0;    // Number of initializers
    size_t parameter_count = 0;      // Total parameters
    double import_time_ms = 0.0;     // Import time in milliseconds
    std::vector<std::string> warnings;  // Warning messages
};
```

#### ImportResult

```cpp
struct ImportResult {
    bool success = false;            // Whether import succeeded
    std::shared_ptr<Graph> graph;    // Imported graph
    std::string error;               // Error message if failed
    ImportStats stats;               // Import statistics
};
```

#### Importer Base Class

```cpp
class Importer {
public:
    virtual ~Importer() = default;

    // Get importer name
    virtual std::string name() const = 0;

    // Get supported file extensions
    virtual std::vector<std::string> supported_extensions() const = 0;

    // Check if file is supported
    virtual bool supports(const std::string& path) const = 0;

    // Import model
    virtual ImportResult import_model(
        const std::string& path,
        const ImportOptions& options = {}) = 0;
};
```

#### ONNXImporter

```cpp
#include <pyflame_rt/import/onnx_importer.hpp>

class ONNXImporter : public Importer {
public:
    ONNXImporter();
    ~ONNXImporter();

    std::string name() const override;
    std::vector<std::string> supported_extensions() const override;
    bool supports(const std::string& path) const override;

    ImportResult import_model(
        const std::string& path,
        const ImportOptions& options = {}) override;

    // Get supported opset versions
    std::vector<int> supported_opsets() const;
};
```

**Example:**

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
    std::cout << "Imported " << result.stats.node_count << " nodes\n";
    std::cout << "Parameters: " << result.stats.parameter_count << "\n";

    // Create session from graph
    InferenceSession session(result.graph);
}
else {
    std::cerr << "Import failed: " << result.error << "\n";
}
```

#### PyTorchImporter

```cpp
#include <pyflame_rt/import/pytorch_importer.hpp>

using ModelDefiner = std::function<void(
    Graph& graph,
    const std::unordered_map<std::string, Tensor>& weights)>;

class PyTorchImporter : public Importer {
public:
    PyTorchImporter();
    ~PyTorchImporter();

    std::string name() const override;
    std::vector<std::string> supported_extensions() const override;
    bool supports(const std::string& path) const override;

    // Standard import (not typically used for PyTorch)
    ImportResult import_model(
        const std::string& path,
        const ImportOptions& options = {}) override;

    // Import with model definer
    ImportResult import_model(
        const std::string& path,
        ModelDefiner definer,
        const std::string& state_dict_key = "",
        const ImportOptions& options = {});
};
```

**Example:**

```cpp
#include <pyflame_rt/import/pytorch_importer.hpp>

using namespace pyflame_rt;

// Define model structure
auto definer = [](Graph& graph,
                  const std::unordered_map<std::string, Tensor>& weights) {
    // Add input
    graph.add_input(TensorInfo{"input", DType::Float32, {{}, 784}});

    // Add weights as initializers
    graph.add_initializer("fc.weight", weights.at("fc.weight"));
    graph.add_initializer("fc.bias", weights.at("fc.bias"));

    // Add node
    auto fc = std::make_shared<Node>("Gemm", "fc",
        std::vector<std::string>{"input", "fc.weight", "fc.bias"},
        std::vector<std::string>{"output"});
    graph.add_node(fc);

    // Add output
    graph.add_output(TensorInfo{"output", DType::Float32, {{}, 10}});
};

// Import
PyTorchImporter importer;
auto result = importer.import_model("model.pth", definer);
```

#### TorchScriptImporter

```cpp
#include <pyflame_rt/import/torchscript_importer.hpp>

class TorchScriptImporter : public Importer {
public:
    TorchScriptImporter();
    ~TorchScriptImporter();

    std::string name() const override;
    std::vector<std::string> supported_extensions() const override;
    bool supports(const std::string& path) const override;

    ImportResult import_model(
        const std::string& path,
        const ImportOptions& options = {}) override;
};
```

#### Importer Registry

```cpp
class ImporterRegistry {
public:
    // Singleton access
    static ImporterRegistry& instance();

    // Register importer
    void register_importer(std::unique_ptr<Importer> importer);

    // Get importer for file
    Importer* get_importer(const std::string& path) const;

    // List registered importers
    std::vector<std::string> list_importers() const;

    // Import using auto-detected format
    ImportResult import_model(
        const std::string& path,
        const ImportOptions& options = {});
};
```

---

### Quantization (C++)

C++ API for model quantization.

#### Namespace

```cpp
namespace pyflame_rt::quantization {
    // Enums
    enum class QuantMode;
    enum class QuantGranularity;
    enum class CalibrationMethod;

    // Types
    struct Float16;
    struct BFloat16;
    struct QuantConfig;
    struct QuantParams;
    struct GraphQuantInfo;
    struct QuantizationResult;

    // Classes
    class Quantizer;
    class Calibrator;
}
```

#### QuantMode

```cpp
enum class QuantMode : uint8_t {
    None = 0,
    FP16 = 1,
    BFloat16 = 2,
    DynamicInt8 = 3,
    StaticInt8 = 4
};

// Helper function
const char* quant_mode_name(QuantMode mode);
```

#### QuantGranularity

```cpp
enum class QuantGranularity : uint8_t {
    PerTensor = 0,
    PerChannel = 1
};
```

#### CalibrationMethod

```cpp
enum class CalibrationMethod : uint8_t {
    MinMax = 0,
    Entropy = 1,
    Percentile = 2
};
```

#### Float16

IEEE 754 half-precision floating-point type.

```cpp
struct Float16 {
    uint16_t bits;

    // Construction
    static Float16 from_float(float value);
    static Float16 from_bits(uint16_t bits);

    // Conversion
    float to_float() const;

    // Special values
    bool is_nan() const;
    bool is_inf() const;
    bool is_zero() const;

    // Arithmetic operators
    Float16 operator+(Float16 other) const;
    Float16 operator-(Float16 other) const;
    Float16 operator*(Float16 other) const;
    Float16 operator/(Float16 other) const;

    // Comparison
    bool operator==(Float16 other) const;
    bool operator<(Float16 other) const;
};
```

#### BFloat16

Brain floating-point type (Google Brain format).

```cpp
struct BFloat16 {
    uint16_t bits;

    // Construction
    static BFloat16 from_float(float value);
    static BFloat16 from_bits(uint16_t bits);

    // Conversion
    float to_float() const;

    // Special values
    bool is_nan() const;
    bool is_inf() const;
    bool is_zero() const;

    // Arithmetic and comparison (same as Float16)
};
```

#### QuantConfig

```cpp
struct QuantConfig {
    QuantMode mode = QuantMode::None;
    DType weight_dtype = DType::Int8;
    DType activation_dtype = DType::Int8;
    QuantGranularity granularity = QuantGranularity::PerTensor;
    CalibrationMethod calibration_method = CalibrationMethod::MinMax;
    bool symmetric = true;
    std::vector<std::string> exclude_ops;
    size_t calibration_samples = 100;

    // Factory methods
    static QuantConfig fp16();
    static QuantConfig bfloat16();
    static QuantConfig dynamic_int8();
    static QuantConfig static_int8(size_t calibration_samples = 100);

    // Validation
    bool is_valid() const;
    std::string validation_error() const;
};
```

#### QuantParams

```cpp
struct QuantParams {
    std::vector<float> scales;
    std::vector<int32_t> zero_points;
    DType quantized_dtype = DType::Int8;
    int channel_axis = -1;  // -1 = per-tensor

    // Factory methods
    static QuantParams per_tensor(float scale, int32_t zero_point,
                                  DType dtype = DType::Int8);
    static QuantParams per_channel(const std::vector<float>& scales,
                                   const std::vector<int32_t>& zero_points,
                                   int channel_axis,
                                   DType dtype = DType::Int8);
    static QuantParams compute_from_minmax(float min_val, float max_val,
                                           DType dtype, bool symmetric);

    // Properties
    bool is_per_channel() const;
    size_t num_channels() const;
};
```

#### GraphQuantInfo

```cpp
struct GraphQuantInfo {
    std::unordered_map<std::string, QuantParams> tensor_params;
    bool weights_quantized = false;
    bool activations_quantized = false;

    // Accessors
    bool has_params(const std::string& tensor_name) const;
    const QuantParams& get_params(const std::string& tensor_name) const;
    void set_params(const std::string& tensor_name, QuantParams params);
};
```

#### QuantizationResult

```cpp
struct QuantizationResult {
    std::unique_ptr<Graph> quantized_graph;
    GraphQuantInfo quant_info;
    bool success = false;
    std::string error_message;

    struct Stats {
        size_t nodes_quantized = 0;
        size_t nodes_skipped = 0;
        size_t weights_quantized = 0;
        size_t original_size_bytes = 0;
        size_t quantized_size_bytes = 0;

        float compression_ratio() const;
        float original_size_mb() const;
        float quantized_size_mb() const;
    } stats;
};
```

#### Quantizer

```cpp
class Quantizer {
public:
    explicit Quantizer(const QuantConfig& config);

    // FP16/BF16 conversion
    QuantizationResult convert_to_fp16(const Graph& graph);
    QuantizationResult convert_to_bfloat16(const Graph& graph);

    // INT8 quantization
    QuantizationResult quantize_dynamic(const Graph& graph);
    QuantizationResult quantize(const Graph& graph,
                                const GraphQuantInfo& quant_info);

    // Convenience method with inline calibration
    using CalibrationDataProvider =
        std::function<std::unordered_map<std::string, Tensor>()>;

    QuantizationResult quantize_with_calibration(
        const Graph& graph,
        CalibrationDataProvider data_provider,
        size_t num_batches);

private:
    QuantConfig config_;
};
```

**Example:**

```cpp
#include <pyflame_rt/quantization/quantizer.hpp>

using namespace pyflame_rt;
using namespace pyflame_rt::quantization;

// FP16 quantization
QuantConfig config = QuantConfig::fp16();
Quantizer quantizer(config);
QuantizationResult result = quantizer.convert_to_fp16(graph);

if (result.success) {
    std::cout << "Compression: " << result.stats.compression_ratio() << "x\n";
    InferenceSession session(std::move(result.quantized_graph));
}

// Dynamic INT8
config = QuantConfig::dynamic_int8();
quantizer = Quantizer(config);
result = quantizer.quantize_dynamic(graph);

// Static INT8 with calibration
config = QuantConfig::static_int8(100);
quantizer = Quantizer(config);

auto data_provider = []() {
    std::unordered_map<std::string, Tensor> data;
    data["input"] = create_random_tensor({1, 3, 224, 224}, DType::Float32);
    return data;
};

result = quantizer.quantize_with_calibration(graph, data_provider, 100);
```

#### Calibrator

```cpp
class Calibrator {
public:
    Calibrator(const Graph& graph, const QuantConfig& config);

    // Feed calibration data
    void observe(const std::unordered_map<std::string, Tensor>& input_feed);

    // Run full calibration
    using CalibrationDataProvider =
        std::function<std::unordered_map<std::string, Tensor>()>;

    void calibrate(CalibrationDataProvider data_provider, size_t num_batches);

    // Compute parameters from collected statistics
    GraphQuantInfo compute_quant_params() const;

    // Get statistics for a specific tensor
    const CalibrationStats& get_stats(const std::string& tensor_name) const;

    // Get all tensor statistics
    const std::unordered_map<std::string, CalibrationStats>& all_stats() const;
};
```

#### CalibrationStats

```cpp
struct CalibrationStats {
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();
    size_t num_samples = 0;
    std::vector<size_t> histogram;

    // Update with new data
    void update(const Tensor& tensor);
    void update(float value);

    // Compute quantization parameters
    QuantParams compute_minmax_params(DType target_dtype, bool symmetric) const;
    QuantParams compute_entropy_params(DType target_dtype, bool symmetric) const;
    QuantParams compute_percentile_params(DType target_dtype, bool symmetric,
                                          float percentile = 99.99f) const;
};
```

**Example:**

```cpp
#include <pyflame_rt/quantization/calibrator.hpp>

using namespace pyflame_rt;
using namespace pyflame_rt::quantization;

// Create calibrator
QuantConfig config = QuantConfig::static_int8();
config.calibration_method = CalibrationMethod::Entropy;
Calibrator calibrator(graph, config);

// Feed calibration data
for (size_t i = 0; i < 100; ++i) {
    auto data = load_calibration_batch(i);
    calibrator.observe(data);
}

// Compute quantization parameters
GraphQuantInfo quant_info = calibrator.compute_quant_params();

// Apply quantization
Quantizer quantizer(config);
QuantizationResult result = quantizer.quantize(graph, quant_info);

// Check statistics
for (const auto& [name, stats] : calibrator.all_stats()) {
    std::cout << name << ": min=" << stats.min_val
              << ", max=" << stats.max_val << "\n";
}
```

---

### Error Handling

```cpp
// Base exception
class PyFlameRTError : public std::runtime_error {
public:
    explicit PyFlameRTError(const std::string& message);
};

// Invalid model
class InvalidModelError : public PyFlameRTError {
public:
    explicit InvalidModelError(const std::string& message);
};

// Unsupported operator
class UnsupportedOperatorError : public PyFlameRTError {
public:
    explicit UnsupportedOperatorError(const std::string& op_type);
    const std::string& op_type() const;
private:
    std::string op_type_;
};

// Validation error
class ValidationError : public PyFlameRTError {
public:
    explicit ValidationError(const std::string& message);
};
```

#### Example: Error Handling

```cpp
#include <pyflame_rt/errors.hpp>
#include <pyflame_rt/session.hpp>

try {
    InferenceSession session("model.pfm");
    auto results = session.run({}, inputs);
}
catch (const InvalidModelError& e) {
    std::cerr << "Invalid model: " << e.what() << "\n";
}
catch (const UnsupportedOperatorError& e) {
    std::cerr << "Unsupported operator: " << e.op_type() << "\n";
}
catch (const ValidationError& e) {
    std::cerr << "Validation failed: " << e.what() << "\n";
}
catch (const PyFlameRTError& e) {
    std::cerr << "PyFlameRT error: " << e.what() << "\n";
}
```

---

## Serving API

The serving module provides infrastructure for deploying PyFlameRT models as HTTP services.

### Python Client

#### ModelClient

Synchronous HTTP client for model inference.

```python
from pyflame_rt.serving import ModelClient

client = ModelClient(
    url: str,                    # Server URL (e.g., "http://localhost:8080")
    timeout: float = 30.0,       # Request timeout in seconds
    max_retries: int = 3,        # Maximum retry attempts
    retry_delay: float = 0.5     # Delay between retries
)
```

##### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `is_alive()` | - | `bool` | Check server liveness |
| `is_ready()` | - | `bool` | Check server readiness |
| `wait_for_ready()` | `timeout: float, poll_interval: float` | `bool` | Wait for server to be ready |
| `list_models()` | - | `List[ModelInfo]` | List all available models |
| `get_model_metadata()` | `model: str` | `ModelMetadata` | Get model metadata |
| `get_model_stats()` | `model: str` | `ModelStats` | Get model statistics |
| `infer()` | `model, inputs, outputs, request_id, priority` | `InferenceResponse` | Run inference |
| `infer_batch()` | `model, batch_inputs, max_workers` | `List[InferenceResponse]` | Batch inference |

##### Example

```python
from pyflame_rt.serving import ModelClient
import numpy as np

client = ModelClient("http://localhost:8080")

# Check health
if client.is_ready():
    # Run inference
    response = client.infer(
        model="resnet50",
        inputs={"input": np.random.randn(1, 3, 224, 224).astype(np.float32)}
    )
    print(f"Output: {response.outputs['output'].shape}")
```

#### AsyncModelClient

Asynchronous HTTP client for model inference. Requires `aiohttp`.

```python
from pyflame_rt.serving import AsyncModelClient

async with AsyncModelClient(
    url: str,                    # Server URL
    timeout: float = 30.0,       # Request timeout
    max_connections: int = 100   # Max concurrent connections
) as client:
    response = await client.infer(model="model", inputs={...})
```

##### Methods

All methods are async versions of `ModelClient` methods.

#### InferenceRequest

Request object for inference.

```python
from pyflame_rt.serving import InferenceRequest

request = InferenceRequest(
    model: str,                           # Model name
    inputs: Dict[str, np.ndarray],        # Input tensors
    outputs: Optional[List[str]] = None,  # Output names (None = all)
    request_id: Optional[str] = None,     # Request ID
    priority: int = 0                     # Priority (higher = more important)
)
```

#### InferenceResponse

Response object from inference.

| Property | Type | Description |
|----------|------|-------------|
| `request_id` | `str` | Request identifier |
| `model_name` | `str` | Model name |
| `model_version` | `str` | Model version |
| `outputs` | `Dict[str, np.ndarray]` | Output tensors |
| `success` | `bool` | Whether inference succeeded |
| `error_message` | `Optional[str]` | Error message if failed |
| `latency_us` | `int` | Latency in microseconds |
| `latency_ms` | `float` | Latency in milliseconds (property) |
| `queue_time_us` | `int` | Queue time in microseconds |

#### ModelInfo

Basic model information.

| Property | Type | Description |
|----------|------|-------------|
| `name` | `str` | Model name |
| `ready` | `bool` | Whether model is ready |
| `versions` | `List[str]` | Available versions |

#### ModelMetadata

Detailed model metadata.

| Property | Type | Description |
|----------|------|-------------|
| `name` | `str` | Model name |
| `version` | `str` | Model version |
| `platform` | `str` | Platform name |
| `ready` | `bool` | Ready status |
| `inputs` | `List[IOSpec]` | Input specifications |
| `outputs` | `List[IOSpec]` | Output specifications |

#### IOSpec

Input/output specification.

| Property | Type | Description |
|----------|------|-------------|
| `name` | `str` | Tensor name |
| `dtype` | `str` | Data type string |
| `shape` | `List[int]` | Tensor shape |

#### ServerError

Exception raised when server returns an error.

| Property | Type | Description |
|----------|------|-------------|
| `status_code` | `int` | HTTP status code |
| `message` | `str` | Error message |

---

### C++ Server

#### ModelServer

Main server class that combines HTTP server, model registry, and metrics.

```cpp
#include "pyflame_rt/serving/model_server.hpp"
using namespace pyflame_rt::serving;

ModelServer(const ServerConfig& config);
```

##### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `start()` | `void` | Start the server |
| `stop()` | `void` | Stop the server |
| `is_running()` | `bool` | Check if server is running |
| `registry()` | `ModelRegistry&` | Get model registry |
| `load_model(config)` | `void` | Load a model |
| `unload_model(name, version)` | `void` | Unload a model |
| `get_stats()` | `ServerStats` | Get server statistics |
| `wait()` | `void` | Block until server stops |
| `http_port()` | `uint16_t` | Get HTTP server port |
| `on_ready(callback)` | `void` | Set ready callback |
| `on_error(callback)` | `void` | Set error callback |

##### Example

```cpp
ServerConfig config;
config.http.port = 8080;
config.enable_metrics = true;

ModelServer server(config);
server.on_ready([]() { std::cout << "Ready!" << std::endl; });
server.start();
server.wait();
```

#### ModelServerBuilder

Fluent builder for creating `ModelServer`.

```cpp
auto server = ModelServerBuilder()
    .host("0.0.0.0")
    .port(8080)
    .workers(4)
    .enable_metrics()
    .add_model("model", "/path/to/model.pfm", "1")
    .enable_batching(32, 5000)
    .build();
```

---

### Configuration Types

#### ServerConfig

Main server configuration.

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `http` | `HTTPServerConfig` | - | HTTP server config |
| `grpc` | `GRPCServerConfig` | - | gRPC server config |
| `tls` | `TLSConfig` | - | TLS configuration |
| `rate_limit` | `RateLimitConfig` | - | Rate limiting config |
| `models` | `vector<ModelConfig>` | `[]` | Models to load |
| `model_dir` | `string` | `""` | Model directory |
| `enable_metrics` | `bool` | `true` | Enable metrics |
| `metrics_port` | `uint16_t` | `9091` | Metrics port |
| `max_memory` | `size_t` | `0` | Memory limit (0=unlimited) |

#### HTTPServerConfig

HTTP server configuration.

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `host` | `string` | `"0.0.0.0"` | Bind address |
| `port` | `uint16_t` | `8080` | Listen port |
| `num_workers` | `size_t` | `0` | Worker threads (0=auto) |
| `max_request_size` | `size_t` | `100MB` | Max request size |
| `request_timeout_ms` | `size_t` | `30000` | Request timeout |
| `keep_alive_timeout_ms` | `size_t` | `5000` | Keep-alive timeout |
| `enable_cors` | `bool` | `false` | Enable CORS |

#### GRPCServerConfig

gRPC server configuration.

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `host` | `string` | `"0.0.0.0"` | Bind address |
| `port` | `uint16_t` | `9090` | Listen port |
| `max_message_size` | `size_t` | `100MB` | Max message size |
| `num_completion_queues` | `size_t` | `0` | Completion queues (0=auto) |

#### TLSConfig

TLS/SSL configuration.

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `enabled` | `bool` | `false` | Enable TLS |
| `cert_path` | `string` | `""` | Certificate path |
| `key_path` | `string` | `""` | Private key path |
| `ca_path` | `string` | `""` | CA certificate path |

#### RateLimitConfig

Rate limiting configuration.

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `enabled` | `bool` | `false` | Enable rate limiting |
| `requests_per_second` | `double` | `100.0` | Requests per second |
| `burst_size` | `size_t` | `50` | Burst size |

#### ModelConfig

Model loading configuration.

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `name` | `string` | `""` | Model name |
| `model_path` | `string` | `""` | Path to model file |
| `version` | `string` | `"1"` | Model version |
| `max_batch_size` | `size_t` | `32` | Max batch size |
| `preferred_batch_sizes` | `vector<size_t>` | `[]` | Preferred sizes |
| `batch_timeout_us` | `size_t` | `5000` | Batch timeout |
| `enable_batching` | `bool` | `false` | Enable batching |
| `warmup_requests` | `size_t` | `3` | Warmup count |

---

### Request/Response Types

#### ServingErrorCode

Error codes for serving operations.

| Value | Description |
|-------|-------------|
| `OK` | Success |
| `InvalidRequest` | Invalid request format |
| `ModelNotFound` | Model not found |
| `ModelNotReady` | Model not ready |
| `InferenceError` | Inference failed |
| `Timeout` | Request timed out |
| `QueueFull` | Request queue full |
| `InternalError` | Internal error |

#### InferRequest (C++)

```cpp
struct InferRequest {
    std::string request_id;
    std::string model_name;
    std::string model_version;
    std::unordered_map<std::string, Tensor> inputs;
    std::vector<std::string> output_names;
    int priority = 0;
    std::chrono::steady_clock::time_point arrival_time;

    static std::string generate_id();
};
```

#### InferResponse (C++)

```cpp
struct InferResponse {
    std::string request_id;
    std::string model_name;
    std::string model_version;
    std::unordered_map<std::string, Tensor> outputs;
    bool success = true;
    ServingErrorCode error_code = ServingErrorCode::OK;
    std::string error_message;
    int64_t latency_us = 0;
    int64_t queue_time_us = 0;
};
```

#### ModelStats

Model statistics.

| Property | Type | Description |
|----------|------|-------------|
| `total_requests` | `uint64_t` | Total request count |
| `successful_requests` | `uint64_t` | Successful requests |
| `failed_requests` | `uint64_t` | Failed requests |
| `avg_latency_ms` | `double` | Average latency |
| `p50_latency_ms` | `double` | P50 latency |
| `p95_latency_ms` | `double` | P95 latency |
| `p99_latency_ms` | `double` | P99 latency |

#### ServingModelMetadata

Model metadata for serving.

| Property | Type | Description |
|----------|------|-------------|
| `name` | `string` | Model name |
| `version` | `string` | Version |
| `platform` | `string` | Platform |
| `inputs` | `vector<IOSpec>` | Input specs |
| `outputs` | `vector<IOSpec>` | Output specs |

---

### Model Registry

#### ModelRegistry

Manages model loading, versioning, and lifecycle.

```cpp
ModelRegistry(size_t max_memory = 0);
```

##### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `register_model(config)` | `void` | Register and load a model |
| `load_from_path(name, path, version)` | `void` | Load model from path |
| `load_from_directory(dir)` | `void` | Load all models from directory |
| `unload(name, version)` | `void` | Unload a model version |
| `unload_all(name)` | `void` | Unload all versions |
| `get(name, version)` | `shared_ptr<ModelInstance>` | Get model instance |
| `get_latest(name)` | `shared_ptr<ModelInstance>` | Get latest version |
| `has(name, version)` | `bool` | Check if model exists |
| `list_models()` | `vector<string>` | List model names |
| `list_versions(name)` | `vector<ModelVersionInfo>` | List versions |
| `enable_hot_reload(enable)` | `void` | Enable hot reload |
| `reload(name, version)` | `void` | Reload a model |
| `memory_used()` | `size_t` | Get memory usage |
| `get_stats()` | `RegistryStats` | Get registry statistics |

#### ModelInstance

Represents a loaded model instance.

##### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `is_ready()` | `bool` | Check if ready |
| `infer(request)` | `InferResponse` | Run inference |
| `infer_async(request)` | `future<InferResponse>` | Async inference |
| `get_stats()` | `ModelStats` | Get statistics |
| `reset_stats()` | `void` | Reset statistics |
| `get_serving_metadata()` | `ServingModelMetadata` | Get metadata |
| `input_names()` | `vector<string>` | Get input names |
| `output_names()` | `vector<string>` | Get output names |
| `config()` | `ModelConfig` | Get configuration |

#### ModelVersionInfo

Version information.

| Property | Type | Description |
|----------|------|-------------|
| `version` | `string` | Version string |
| `path` | `string` | Model path |
| `is_loaded` | `bool` | Load status |

---

### Metrics

#### MetricsRegistry

Singleton registry for Prometheus metrics.

```cpp
static MetricsRegistry& instance();
```

##### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `counter_inc(name, labels...)` | `void` | Increment counter |
| `gauge_set(name, value, labels...)` | `void` | Set gauge value |
| `histogram_observe(name, value, labels...)` | `void` | Record histogram |
| `export_prometheus()` | `string` | Export metrics |
| `reset()` | `void` | Reset all metrics |

#### Convenience Functions

```cpp
namespace metrics {
    void request_total(const std::string& model, const std::string& status);
    void request_latency(const std::string& model, double seconds);
    void inference_error(const std::string& model, const std::string& error_type);
    void model_loaded(const std::string& model, bool loaded);
    void batch_size(const std::string& model, size_t size);
    void queue_size(const std::string& model, size_t size);
    void request_active_inc(const std::string& model);
    void request_active_dec(const std::string& model);
}
```

#### Available Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `pyflame_request_total` | Counter | model, status | Total requests |
| `pyflame_request_latency_seconds` | Histogram | model | Latency distribution |
| `pyflame_requests_active` | Gauge | model | Active requests |
| `pyflame_model_loaded` | Gauge | model | Model status |
| `pyflame_batch_size` | Histogram | model | Batch sizes |
| `pyflame_queue_size` | Gauge | model | Queue depth |
| `pyflame_inference_errors_total` | Counter | model, error | Error counts |

---

## Appendix: Type Conversions

### NumPy to DType Mapping

| NumPy dtype | DType |
|-------------|-------|
| `np.float32` | `DType::Float32` |
| `np.float64` | `DType::Float64` |
| `np.float16` | `DType::Float16` |
| `np.int64` | `DType::Int64` |
| `np.int32` | `DType::Int32` |
| `np.int16` | `DType::Int16` |
| `np.int8` | `DType::Int8` |
| `np.uint8` | `DType::UInt8` |
| `np.bool_` | `DType::Bool` |

### C++ Type to DType Mapping

| C++ Type | DType |
|----------|-------|
| `float` | `DType::Float32` |
| `double` | `DType::Float64` |
| `int64_t` | `DType::Int64` |
| `int32_t` | `DType::Int32` |
| `int16_t` | `DType::Int16` |
| `int8_t` | `DType::Int8` |
| `uint8_t` | `DType::UInt8` |
| `bool` | `DType::Bool` |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.6.0 | 2026 | Added Serving Infrastructure (HTTP server, model registry, metrics, K8s) |
| 0.5.0 | 2026 | Added Production Features (memory pool, caching, batching, streaming) |
| 0.4.0 | 2026 | Added Quantization API (FP16, BFloat16, INT8 dynamic/static) |
| 0.3.0 | 2026 | Added Graph Optimization API (PassManager, fusion, folding) |
| 0.2.0 | 2026 | Added Model Import API (ONNX, PyTorch, TorchScript) |
| 0.1.0 | 2026 | Initial release |
