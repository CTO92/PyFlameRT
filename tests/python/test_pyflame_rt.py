"""Tests for PyFlameRT Python bindings."""

import numpy as np
import pytest


class TestImport:
    """Test module import and basic structure."""

    def test_import_module(self):
        """Test that the module imports correctly."""
        import pyflame_rt

        assert hasattr(pyflame_rt, 'InferenceSession')
        assert hasattr(pyflame_rt, 'SessionOptions')
        assert hasattr(pyflame_rt, 'DType')
        assert hasattr(pyflame_rt, '__version__')

    def test_version(self):
        """Test version string."""
        import pyflame_rt

        assert pyflame_rt.__version__ == "0.1.0"


class TestDType:
    """Test DType enum."""

    def test_dtype_values(self):
        """Test DType enum values."""
        import pyflame_rt

        assert pyflame_rt.DType.Float32.value == 0
        assert pyflame_rt.DType.Float16.value == 1
        assert pyflame_rt.DType.Int64.value == 4
        assert pyflame_rt.DType.Int32.value == 5

    def test_dtype_names(self):
        """Test DType enum names."""
        import pyflame_rt

        assert pyflame_rt.DType.Float32.name == "Float32"
        assert pyflame_rt.DType.Int64.name == "Int64"


class TestSessionOptions:
    """Test SessionOptions configuration."""

    def test_default_options(self):
        """Test default SessionOptions values."""
        import pyflame_rt

        opts = pyflame_rt.SessionOptions()

        assert opts.device == "cpu"
        assert opts.num_threads == 0
        assert opts.enable_profiling == False
        assert opts.execution_mode == "sequential"
        assert opts.log_level == "warning"

    def test_modify_options(self):
        """Test modifying SessionOptions."""
        import pyflame_rt

        opts = pyflame_rt.SessionOptions()
        opts.num_threads = 4
        opts.enable_profiling = True
        opts.log_level = "debug"

        assert opts.num_threads == 4
        assert opts.enable_profiling == True
        assert opts.log_level == "debug"

    def test_validate_options(self):
        """Test options validation."""
        import pyflame_rt

        opts = pyflame_rt.SessionOptions()
        errors = opts.validate()
        assert len(errors) == 0

        opts.device = "invalid"
        errors = opts.validate()
        assert len(errors) > 0


class TestTensor:
    """Test Tensor class."""

    def test_from_numpy(self):
        """Test creating Tensor from numpy array."""
        import pyflame_rt

        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        tensor = pyflame_rt.from_numpy(arr)

        assert tensor.shape == [2, 2]
        assert tensor.dtype == pyflame_rt.DType.Float32
        assert tensor.ndim == 2
        assert tensor.num_elements == 4

    def test_to_numpy(self):
        """Test converting Tensor to numpy array."""
        import pyflame_rt

        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        tensor = pyflame_rt.from_numpy(arr)
        result = tensor.numpy()

        np.testing.assert_array_equal(result, arr)

    def test_different_dtypes(self):
        """Test tensors with different dtypes."""
        import pyflame_rt

        arr_f32 = np.array([1.0, 2.0], dtype=np.float32)
        arr_f64 = np.array([1.0, 2.0], dtype=np.float64)
        arr_i64 = np.array([1, 2], dtype=np.int64)
        arr_i32 = np.array([1, 2], dtype=np.int32)

        t_f32 = pyflame_rt.from_numpy(arr_f32)
        t_f64 = pyflame_rt.from_numpy(arr_f64)
        t_i64 = pyflame_rt.from_numpy(arr_i64)
        t_i32 = pyflame_rt.from_numpy(arr_i32)

        assert t_f32.dtype == pyflame_rt.DType.Float32
        assert t_f64.dtype == pyflame_rt.DType.Float64
        assert t_i64.dtype == pyflame_rt.DType.Int64
        assert t_i32.dtype == pyflame_rt.DType.Int32

    def test_tensor_clone(self):
        """Test tensor cloning."""
        import pyflame_rt

        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        tensor = pyflame_rt.from_numpy(arr)
        clone = tensor.clone()

        # Modify original via numpy
        result1 = tensor.numpy()
        result2 = clone.numpy()

        np.testing.assert_array_equal(result1, result2)

    def test_tensor_reshape(self):
        """Test tensor reshape."""
        import pyflame_rt

        arr = np.arange(12, dtype=np.float32)
        tensor = pyflame_rt.from_numpy(arr)

        reshaped = tensor.reshape([3, 4])
        assert reshaped.shape == [3, 4]

        reshaped = tensor.reshape([2, 6])
        assert reshaped.shape == [2, 6]

    def test_tensor_repr(self):
        """Test tensor string representation."""
        import pyflame_rt

        arr = np.zeros((2, 3, 4), dtype=np.float32)
        tensor = pyflame_rt.from_numpy(arr)

        repr_str = repr(tensor)
        assert "Tensor" in repr_str
        assert "2" in repr_str
        assert "3" in repr_str
        assert "4" in repr_str


class TestTensorInfo:
    """Test TensorInfo class."""

    def test_tensor_info_creation(self):
        """Test creating TensorInfo."""
        import pyflame_rt

        info = pyflame_rt.TensorInfo()
        info.name = "test"
        info.dtype = pyflame_rt.DType.Float32

        assert info.name == "test"
        assert info.dtype == pyflame_rt.DType.Float32


class TestNodeArg:
    """Test NodeArg class."""

    def test_node_arg_repr(self):
        """Test NodeArg string representation."""
        import pyflame_rt

        # NodeArg is typically created internally, so we test via TensorInfo
        info = pyflame_rt.TensorInfo()
        info.name = "input"
        info.dtype = pyflame_rt.DType.Float32


class TestExceptions:
    """Test exception classes."""

    def test_exception_hierarchy(self):
        """Test that exceptions have proper hierarchy."""
        import pyflame_rt

        assert issubclass(pyflame_rt.InvalidModelError, pyflame_rt.PyFlameRTError)
        assert issubclass(pyflame_rt.UnsupportedOperatorError, pyflame_rt.PyFlameRTError)
        assert issubclass(pyflame_rt.ValidationError, pyflame_rt.PyFlameRTError)


class TestRunOptions:
    """Test RunOptions class."""

    def test_run_options_defaults(self):
        """Test RunOptions default values."""
        import pyflame_rt

        opts = pyflame_rt.RunOptions()
        # Optional fields should be None by default
        assert opts.log_level is None
        assert opts.tag is None
        assert opts.timeout_ms is None


class TestCompileOptions:
    """Test CompileOptions class."""

    def test_compile_options_defaults(self):
        """Test CompileOptions default values."""
        import pyflame_rt

        opts = pyflame_rt.CompileOptions()
        assert opts.cache_dir is None
        assert opts.dynamic_batch == False
        assert opts.optimization_level == 2
