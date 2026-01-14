#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "pyflame_rt/tensor.hpp"
#include "pyflame_rt/types.hpp"
#include <stdexcept>

namespace py = pybind11;
using namespace pyflame_rt;

namespace {

// Security: Maximum tensor size to prevent resource exhaustion (MED-06 fix)
constexpr size_t MAX_TENSOR_BYTES = 16ULL * 1024 * 1024 * 1024;  // 16 GB

// Convert numpy dtype to DType
DType numpy_to_dtype(py::dtype dt) {
    if (dt.is(py::dtype::of<float>())) return DType::Float32;
    if (dt.is(py::dtype::of<double>())) return DType::Float64;
    if (dt.is(py::dtype::of<int64_t>())) return DType::Int64;
    if (dt.is(py::dtype::of<int32_t>())) return DType::Int32;
    if (dt.is(py::dtype::of<int16_t>())) return DType::Int16;
    if (dt.is(py::dtype::of<int8_t>())) return DType::Int8;
    if (dt.is(py::dtype::of<uint8_t>())) return DType::UInt8;
    if (dt.is(py::dtype::of<bool>())) return DType::Bool;
    throw std::runtime_error("Unsupported numpy dtype");
}

// Convert DType to numpy dtype string
std::string dtype_to_numpy_str(DType dt) {
    switch (dt) {
        case DType::Float32: return "float32";
        case DType::Float64: return "float64";
        case DType::Float16: return "float16";
        case DType::Int64: return "int64";
        case DType::Int32: return "int32";
        case DType::Int16: return "int16";
        case DType::Int8: return "int8";
        case DType::UInt8: return "uint8";
        case DType::Bool: return "bool";
        default: return "float32";
    }
}

} // anonymous namespace

/// Create Tensor from numpy array with validation (MED-01 fix)
Tensor tensor_from_numpy(py::array arr) {
    // Security: ensure array is contiguous and check for failure (MED-01)
    py::array contiguous = py::array::ensure(arr, py::array::c_style);
    if (!contiguous) {
        throw std::runtime_error(
            "Failed to convert numpy array to contiguous C-style array");
    }

    // Security: validate array has data
    if (contiguous.data() == nullptr && contiguous.size() > 0) {
        throw std::runtime_error("Numpy array has null data pointer");
    }

    // Get shape with validation
    std::vector<int64_t> shape;
    shape.reserve(static_cast<size_t>(contiguous.ndim()));
    for (py::ssize_t i = 0; i < contiguous.ndim(); ++i) {
        py::ssize_t dim = contiguous.shape(i);
        if (dim < 0) {
            throw std::runtime_error(
                "Invalid negative dimension in numpy array at axis " + std::to_string(i));
        }
        shape.push_back(static_cast<int64_t>(dim));
    }

    DType dtype = numpy_to_dtype(contiguous.dtype());

    // Security fix MED-06: Enforce maximum tensor size to prevent resource exhaustion
    size_t numpy_bytes = static_cast<size_t>(contiguous.nbytes());
    if (numpy_bytes > MAX_TENSOR_BYTES) {
        throw std::runtime_error(
            "Tensor size " + std::to_string(numpy_bytes / (1024 * 1024)) +
            " MB exceeds maximum allowed size of " +
            std::to_string(MAX_TENSOR_BYTES / (1024 * 1024 * 1024)) + " GB");
    }

    Tensor tensor(shape, dtype);

    // Security: validate sizes match before copy (MED-01)
    size_t tensor_bytes = tensor.size_bytes();
    if (numpy_bytes != tensor_bytes) {
        throw std::runtime_error(
            "Size mismatch: numpy array has " + std::to_string(numpy_bytes) +
            " bytes, tensor expects " + std::to_string(tensor_bytes) + " bytes");
    }

    // Safe to copy now
    if (tensor_bytes > 0) {
        std::memcpy(tensor.data(), contiguous.data(), tensor_bytes);
    }

    return tensor;
}

/// Create numpy array from Tensor with validation (MED-01 fix)
py::array tensor_to_numpy(const Tensor& tensor) {
    // Security: validate tensor before conversion
    if (!tensor.is_valid() && tensor.num_elements() > 0) {
        throw std::runtime_error("Cannot convert invalid tensor to numpy array");
    }

    std::string dtype_str = dtype_to_numpy_str(tensor.dtype());

    // Create numpy array
    std::vector<py::ssize_t> shape(tensor.shape().begin(), tensor.shape().end());
    py::array result(py::dtype(dtype_str), shape);

    // Security: validate sizes match
    size_t tensor_bytes = tensor.size_bytes();
    size_t numpy_bytes = static_cast<size_t>(result.nbytes());
    if (tensor_bytes != numpy_bytes) {
        throw std::runtime_error(
            "Size mismatch during tensor-to-numpy conversion");
    }

    // Safe to copy
    if (tensor_bytes > 0 && tensor.data() != nullptr) {
        std::memcpy(result.mutable_data(), tensor.data(), tensor_bytes);
    }

    return result;
}

void bind_tensor(py::module_& m) {
    py::class_<Tensor>(m, "Tensor", "Tensor data container")
        .def(py::init([](py::array arr) {
            return tensor_from_numpy(arr);
        }), py::arg("data"), "Create tensor from numpy array")

        .def(py::init([](const std::vector<int64_t>& shape, DType dtype) {
            return Tensor(shape, dtype);
        }), py::arg("shape"), py::arg("dtype") = DType::Float32,
           "Create tensor with given shape and dtype")

        .def_property_readonly("shape", [](const Tensor& t) {
            return std::vector<int64_t>(t.shape().begin(), t.shape().end());
        }, "Tensor shape")

        .def_property_readonly("dtype", &Tensor::dtype, "Data type")
        .def_property_readonly("ndim", &Tensor::ndim, "Number of dimensions")
        .def_property_readonly("num_elements", &Tensor::num_elements, "Total elements")
        .def_property_readonly("size_bytes", &Tensor::size_bytes, "Size in bytes")
        .def_property_readonly("is_valid", &Tensor::is_valid, "Check if tensor is valid")
        .def_property_readonly("owns_data", &Tensor::owns_data, "Check if tensor owns data")

        .def("numpy", &tensor_to_numpy, "Convert to numpy array")
        .def("clone", &Tensor::clone, "Create a deep copy")
        .def("zero", &Tensor::zero, "Fill tensor with zeros")

        .def("reshape", &Tensor::reshape, py::arg("shape"),
             "Reshape tensor (must have same total elements)")

        .def("__repr__", [](const Tensor& t) {
            std::string repr = "Tensor(shape=[";
            for (size_t i = 0; i < t.shape().size(); ++i) {
                if (i > 0) repr += ", ";
                repr += std::to_string(t.shape()[i]);
            }
            repr += "], dtype=" + dtype_name(t.dtype()) + ")";
            return repr;
        });

    // Module-level conversion functions
    m.def("from_numpy", &tensor_from_numpy,
          py::arg("array"),
          "Create Tensor from numpy array");

    m.def("to_numpy", &tensor_to_numpy,
          py::arg("tensor"),
          "Convert Tensor to numpy array");
}
