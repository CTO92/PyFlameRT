"""
PyFlameRT - High-performance inference runtime for Cerebras WSE.

This module provides the main public API for PyFlameRT.
"""

try:
    from ._pyflame_rt import (
        # Main session class
        InferenceSession,
        create_session,

        # Options
        SessionOptions,
        RunOptions,
        CompileOptions,

        # Types
        DType,
        OptLevel,
        TensorInfo,
        NodeArg,
        ModelMetadata,
        Tensor,

        # Conversion functions
        from_numpy,
        to_numpy,

        # ONNX import convenience function
        from_onnx,

        # Exceptions
        PyFlameRTError,
        InvalidModelError,
        UnsupportedFormatError,
        UnsupportedOperatorError,
        ShapeMismatchError,
        DTypeMismatchError,
        ValidationError,
        BackendError,
        InputError,

        # Version
        __version__,

        # Import submodule
        import_ as import_module,

        # Optimization submodule
        opt,

        # Quantization submodule
        quantization,

        # Phase 5: Production features
        cache,
        memory,
        batching,
        streaming,

        # Phase 7: Advanced optimization
        pruning,
        distillation,
        custom,
        partition,
    )

    # Phase 6: Serving (optional, may not be built)
    try:
        from ._pyflame_rt import serving
    except ImportError:
        serving = None

    # Also import pure Python serving client
    from . import serving as serving_client

except ImportError as e:
    raise ImportError(
        "Failed to import PyFlameRT C++ extension. "
        "Make sure the library is properly built.\n"
        f"Original error: {e}"
    ) from e

__all__ = [
    # Main classes
    "InferenceSession",
    "create_session",
    "SessionOptions",
    "RunOptions",
    "CompileOptions",

    # Types
    "DType",
    "OptLevel",
    "TensorInfo",
    "NodeArg",
    "ModelMetadata",
    "Tensor",

    # Conversion
    "from_numpy",
    "to_numpy",

    # Import functionality
    "from_onnx",
    "import_module",

    # Optimization
    "opt",

    # Quantization
    "quantization",

    # Phase 5: Production features
    "cache",
    "memory",
    "batching",
    "streaming",

    # Phase 6: Serving
    "serving",
    "serving_client",

    # Phase 7: Advanced optimization
    "pruning",
    "distillation",
    "custom",
    "partition",

    # Exceptions
    "PyFlameRTError",
    "InvalidModelError",
    "UnsupportedFormatError",
    "UnsupportedOperatorError",
    "ShapeMismatchError",
    "DTypeMismatchError",
    "ValidationError",
    "BackendError",
    "InputError",

    # Version
    "__version__",
]
