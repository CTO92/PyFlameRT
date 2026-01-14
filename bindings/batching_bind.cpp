#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/chrono.h>
#include <pybind11/functional.h>
#include "pyflame_rt/batching/dynamic_batcher.hpp"

namespace py = pybind11;
using namespace pyflame_rt;
using namespace pyflame_rt::batching;

// Forward declaration from tensor_bind.cpp
Tensor tensor_from_numpy(py::array arr);
py::array tensor_to_numpy(const Tensor& tensor);

void bind_batching(py::module_& m) {
    auto batch_mod = m.def_submodule("batching", "Dynamic batching support");

    // ========================================================================
    // BatchConfig
    // ========================================================================
    py::class_<BatchConfig>(batch_mod, "BatchConfig",
        "Configuration for dynamic batching.")
        .def(py::init<>())
        .def_readwrite("max_batch_size", &BatchConfig::max_batch_size,
            "Maximum batch size")
        .def_readwrite("max_latency", &BatchConfig::max_latency,
            "Maximum time to wait for batch to fill")
        .def_readwrite("min_batch_size", &BatchConfig::min_batch_size,
            "Minimum batch size before executing")
        .def_readwrite("preferred_sizes", &BatchConfig::preferred_sizes,
            "Preferred batch sizes for padding")
        .def_readwrite("enable_padding", &BatchConfig::enable_padding,
            "Enable padding to preferred sizes")
        .def_readwrite("num_workers", &BatchConfig::num_workers,
            "Number of worker threads (0 = auto)")
        .def_readwrite("queue_capacity", &BatchConfig::queue_capacity,
            "Queue capacity (0 = unlimited)")
        .def_readwrite("queue_timeout", &BatchConfig::queue_timeout,
            "Timeout for queue operations");

    // ========================================================================
    // InferenceRequest
    // ========================================================================
    py::class_<InferenceRequest>(batch_mod, "InferenceRequest",
        "Single inference request.")
        .def(py::init<>())
        .def_readwrite("id", &InferenceRequest::id,
            "Request ID (assigned by batcher)")
        .def_readwrite("output_names", &InferenceRequest::output_names,
            "Requested output names (empty = all)")
        .def_readwrite("priority", &InferenceRequest::priority,
            "Priority (higher = process sooner)")
        .def("set_inputs", [](InferenceRequest& self, const py::dict& inputs) {
            for (auto item : inputs) {
                std::string name = py::str(item.first);
                py::array arr = py::array::ensure(item.second);
                self.inputs[name] = tensor_from_numpy(arr);
            }
        }, py::arg("inputs"),
           "Set input tensors from numpy dictionary");

    // ========================================================================
    // InferenceResponse
    // ========================================================================
    py::class_<InferenceResponse>(batch_mod, "InferenceResponse",
        "Inference response.")
        .def_readonly("request_id", &InferenceResponse::request_id,
            "Request ID")
        .def_readonly("latency", &InferenceResponse::latency,
            "Processing latency")
        .def_readonly("success", &InferenceResponse::success,
            "Whether inference succeeded")
        .def_readonly("error_message", &InferenceResponse::error_message,
            "Error message if failed")
        .def("get_outputs", [](const InferenceResponse& self) {
            py::dict result;
            for (const auto& [name, tensor] : self.outputs) {
                result[py::str(name)] = tensor_to_numpy(tensor);
            }
            return result;
        }, "Get output tensors as numpy dictionary");

    // ========================================================================
    // DynamicBatcher::Stats
    // ========================================================================
    py::class_<DynamicBatcher::Stats>(batch_mod, "BatcherStats",
        "Batcher statistics.")
        .def_readonly("total_requests", &DynamicBatcher::Stats::total_requests,
            "Total requests processed")
        .def_readonly("total_batches", &DynamicBatcher::Stats::total_batches,
            "Total batches processed")
        .def_readonly("dropped_requests", &DynamicBatcher::Stats::dropped_requests,
            "Number of dropped requests")
        .def_readonly("avg_batch_size", &DynamicBatcher::Stats::avg_batch_size,
            "Average batch size")
        .def_readonly("avg_latency_us", &DynamicBatcher::Stats::avg_latency_us,
            "Average latency in microseconds")
        .def_readonly("throughput_rps", &DynamicBatcher::Stats::throughput_rps,
            "Throughput in requests per second")
        .def_readonly("queue_depth", &DynamicBatcher::Stats::queue_depth,
            "Current queue depth");

    // ========================================================================
    // DynamicBatcher
    // ========================================================================
    py::class_<DynamicBatcher>(batch_mod, "DynamicBatcher",
        "Dynamic batcher for automatic request batching.\n"
        "Batches incoming inference requests to maximize throughput.")
        .def(py::init<std::shared_ptr<InferenceSession>, const BatchConfig&>(),
             py::arg("session"),
             py::arg("config") = BatchConfig(),
             "Create a dynamic batcher.")

        .def("submit", [](DynamicBatcher& self, const py::dict& inputs,
                         std::optional<std::vector<std::string>> output_names) {
            InferenceRequest request;
            for (auto item : inputs) {
                std::string name = py::str(item.first);
                py::array arr = py::array::ensure(item.second);
                request.inputs[name] = tensor_from_numpy(arr);
            }
            if (output_names) {
                request.output_names = *output_names;
            }
            return self.submit(std::move(request));
        },
             py::arg("inputs"),
             py::arg("output_names") = py::none(),
             "Submit an inference request asynchronously.")

        .def("submit_with_callback", [](DynamicBatcher& self, const py::dict& inputs,
                                        py::function callback,
                                        std::optional<std::vector<std::string>> output_names) {
            InferenceRequest request;
            for (auto item : inputs) {
                std::string name = py::str(item.first);
                py::array arr = py::array::ensure(item.second);
                request.inputs[name] = tensor_from_numpy(arr);
            }
            if (output_names) {
                request.output_names = *output_names;
            }
            return self.submit(std::move(request), [callback](InferenceResponse response) {
                py::gil_scoped_acquire acquire;
                py::dict outputs;
                for (const auto& [name, tensor] : response.outputs) {
                    outputs[py::str(name)] = tensor_to_numpy(tensor);
                }
                callback(response.request_id, response.success, outputs, response.error_message);
            });
        },
             py::arg("inputs"),
             py::arg("callback"),
             py::arg("output_names") = py::none(),
             "Submit request with callback.")

        .def("infer", [](DynamicBatcher& self, const py::dict& inputs,
                        std::optional<std::vector<std::string>> output_names) {
            InferenceRequest request;
            for (auto item : inputs) {
                std::string name = py::str(item.first);
                py::array arr = py::array::ensure(item.second);
                request.inputs[name] = tensor_from_numpy(arr);
            }
            if (output_names) {
                request.output_names = *output_names;
            }
            auto response = self.infer(std::move(request));
            py::dict result;
            for (const auto& [name, tensor] : response.outputs) {
                result[py::str(name)] = tensor_to_numpy(tensor);
            }
            return result;
        },
             py::arg("inputs"),
             py::arg("output_names") = py::none(),
             "Blocking inference (waits for result).")

        .def("start", &DynamicBatcher::start,
             "Start the batcher.")
        .def("stop", &DynamicBatcher::stop,
             "Stop the batcher.")
        .def("is_running", &DynamicBatcher::is_running,
             "Check if batcher is running.")
        .def("flush", &DynamicBatcher::flush,
             "Flush pending requests.")
        .def("queue_size", &DynamicBatcher::queue_size,
             "Get current queue size.")
        .def("get_stats", &DynamicBatcher::get_stats,
             "Get batcher statistics.")
        .def("reset_stats", &DynamicBatcher::reset_stats,
             "Reset statistics.")
        .def("update_config", &DynamicBatcher::update_config,
             py::arg("config"),
             "Update batcher configuration.")
        .def_property_readonly("config", &DynamicBatcher::config,
             "Get batcher configuration.");

    // ========================================================================
    // PriorityBatcher
    // ========================================================================
    py::class_<PriorityBatcher, DynamicBatcher>(batch_mod, "PriorityBatcher",
        "Priority queue batcher.\n"
        "Processes high-priority requests first.")
        .def(py::init<std::shared_ptr<InferenceSession>, const BatchConfig&>(),
             py::arg("session"),
             py::arg("config") = BatchConfig(),
             "Create a priority batcher.")
        .def("submit_priority", [](PriorityBatcher& self, const py::dict& inputs,
                                   int priority,
                                   std::optional<std::vector<std::string>> output_names) {
            InferenceRequest request;
            for (auto item : inputs) {
                std::string name = py::str(item.first);
                py::array arr = py::array::ensure(item.second);
                request.inputs[name] = tensor_from_numpy(arr);
            }
            if (output_names) {
                request.output_names = *output_names;
            }
            return self.submit_priority(std::move(request), priority);
        },
             py::arg("inputs"),
             py::arg("priority"),
             py::arg("output_names") = py::none(),
             "Submit a high-priority request.");

    // ========================================================================
    // Factory functions
    // ========================================================================
    batch_mod.def("create_batcher", py::overload_cast<std::shared_ptr<InferenceSession>>(
        &create_batcher),
        py::arg("session"),
        "Create a batcher with default configuration.");

    batch_mod.def("create_batcher", py::overload_cast<std::shared_ptr<InferenceSession>,
        const BatchConfig&>(&create_batcher),
        py::arg("session"),
        py::arg("config"),
        "Create a batcher with custom configuration.");
}
