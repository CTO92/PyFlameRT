"""pytest configuration for PyFlameRT tests."""

import pytest
import numpy as np
import tempfile
import os


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_input():
    """Create sample input data for testing."""
    return np.random.randn(1, 3, 224, 224).astype(np.float32)


@pytest.fixture
def simple_model_path(temp_dir):
    """
    Create a simple test model file.

    Note: This requires the C++ library to be built and accessible.
    For now, this is a placeholder that tests can use once the library is built.
    """
    model_path = os.path.join(temp_dir, "test_model.pfm")
    # Model creation would go here once the library is fully integrated
    return model_path
