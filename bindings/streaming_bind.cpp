#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/chrono.h>
#include <pybind11/functional.h>
#include "pyflame_rt/streaming/async_session.hpp"

namespace py = pybind11;
using namespace pyflame_rt;
using namespace pyflame_rt::streaming;

// Forward declaration from tensor_bind.cpp
Tensor tensor_from_numpy(py::array arr);
py::array tensor_to_numpy(const Tensor& tensor);

void bind_streaming(py::module_& m) {
    auto stream_mod = m.def_submodule("streaming", "Asynchronous and streaming inference");

    // ========================================================================
    // AsyncOptions
    // ========================================================================
    py::class_<AsyncOptions>(stream_mod, "AsyncOptions",
        "Options for asynchronous inference.")
        .def(py::init<>())
        .def_readwrite("num_streams", &AsyncOptions::num_streams,
            "Number of concurrent inference streams")
        .def_readwrite("enable_pipelining", &AsyncOptions::enable_pipelining,
            "Enable pipelining")
        .def_readwrite("callback_threads", &AsyncOptions::callback_threads,
            "Callback thread pool size")
        .def_readwrite("max_pending", &AsyncOptions::max_pending,
            "Maximum pending requests per stream")
        .def_readwrite("enable_profiling", &AsyncOptions::enable_profiling,
            "Enable profiling for async operations");

    // ========================================================================
    // StreamContext
    // ========================================================================
    py::class_<StreamContext>(stream_mod, "StreamContext",
        "Stream context for managing inference state.")
        .def("stream_id", &StreamContext::stream_id,
             "Get stream ID")
        .def("is_busy", &StreamContext::is_busy,
             "Check if stream is busy")
        .def("pending_count", &StreamContext::pending_count,
             "Get number of pending operations")
        .def("synchronize", &StreamContext::synchronize,
             "Wait for all pending operations")
        .def("record_event", &StreamContext::record_event,
             py::arg("name"),
             "Record event in stream")
        .def("wait_event", &StreamContext::wait_event,
             py::arg("name"),
             "Wait for event");

    // ========================================================================
    // AsyncResult::ProfilingData
    // ========================================================================
    py::class_<AsyncResult::ProfilingData>(stream_mod, "ProfilingData",
        "Profiling data for async result.")
        .def_readonly("queue_time", &AsyncResult::ProfilingData::queue_time,
            "Time spent in queue")
        .def_readonly("compute_time", &AsyncResult::ProfilingData::compute_time,
            "Compute time")
        .def_readonly("callback_time", &AsyncResult::ProfilingData::callback_time,
            "Callback execution time");

    // ========================================================================
    // AsyncResult
    // ========================================================================
    py::class_<AsyncResult>(stream_mod, "AsyncResult",
        "Async inference result.")
        .def_readonly("request_id", &AsyncResult::request_id,
            "Request ID")
        .def_readonly("stream_id", &AsyncResult::stream_id,
            "Stream used for inference")
        .def_readonly("latency", &AsyncResult::latency,
            "Inference latency")
        .def_readonly("success", &AsyncResult::success,
            "Success status")
        .def_readonly("error_message", &AsyncResult::error_message,
            "Error message if failed")
        .def_property_readonly("profiling", [](const AsyncResult& self) {
            return self.profiling;
        }, "Profiling data (if enabled)")
        .def("get_outputs", [](const AsyncResult& self) {
            py::dict result;
            for (const auto& [name, tensor] : self.outputs) {
                result[py::str(name)] = tensor_to_numpy(tensor);
            }
            return result;
        }, "Get output tensors as numpy dictionary");

    // ========================================================================
    // AsyncSession::Stats
    // ========================================================================
    py::class_<AsyncSession::Stats>(stream_mod, "AsyncStats",
        "Async session statistics.")
        .def_readonly("total_requests", &AsyncSession::Stats::total_requests,
            "Total requests submitted")
        .def_readonly("completed_requests", &AsyncSession::Stats::completed_requests,
            "Completed requests")
        .def_readonly("failed_requests", &AsyncSession::Stats::failed_requests,
            "Failed requests")
        .def_readonly("cancelled_requests", &AsyncSession::Stats::cancelled_requests,
            "Cancelled requests")
        .def_readonly("avg_latency_us", &AsyncSession::Stats::avg_latency_us,
            "Average latency in microseconds")
        .def_readonly("avg_queue_time_us", &AsyncSession::Stats::avg_queue_time_us,
            "Average queue time in microseconds")
        .def_readonly("stream_utilization", &AsyncSession::Stats::stream_utilization,
            "Stream utilization (pending count per stream)");

    // ========================================================================
    // AsyncSession
    // ========================================================================
    py::class_<AsyncSession>(stream_mod, "AsyncSession",
        "Asynchronous inference session.\n"
        "Provides non-blocking inference with multiple streams.")
        .def(py::init<std::shared_ptr<InferenceSession>, const AsyncOptions&>(),
             py::arg("session"),
             py::arg("options") = AsyncOptions(),
             "Create an async session.")

        .def("run_async", [](AsyncSession& self, const py::dict& inputs,
                            std::optional<std::vector<std::string>> output_names) {
            std::unordered_map<std::string, Tensor> feed;
            for (auto item : inputs) {
                std::string name = py::str(item.first);
                // Security: validate array conversion (CRIT-B2 fix)
                py::array arr = py::array::ensure(item.second);
                if (!arr) {
                    throw std::runtime_error(
                        "Input '" + name + "' could not be converted to numpy array");
                }
                feed[name] = tensor_from_numpy(arr);
            }
            return self.run_async(output_names.value_or(std::vector<std::string>{}), feed);
        },
             py::arg("inputs"),
             py::arg("output_names") = py::none(),
             "Submit async inference request.")

        .def("run_async_callback", [](AsyncSession& self, const py::dict& inputs,
                                      py::function callback,
                                      std::optional<std::vector<std::string>> output_names) {
            std::unordered_map<std::string, Tensor> feed;
            for (auto item : inputs) {
                std::string name = py::str(item.first);
                // Security: validate array conversion (HIGH-B1 fix)
                py::array arr = py::array::ensure(item.second);
                if (!arr) {
                    throw std::runtime_error(
                        "Input '" + name + "' could not be converted to numpy array");
                }
                feed[name] = tensor_from_numpy(arr);
            }
            // Security: copy callback to ensure lifetime (HIGH-B2 fix)
            py::function callback_copy = callback;
            return self.run_async(
                output_names.value_or(std::vector<std::string>{}),
                feed,
                [callback_copy](AsyncResult result) {
                    py::gil_scoped_acquire acquire;
                    try {
                        py::dict outputs;
                        for (const auto& [name, tensor] : result.outputs) {
                            outputs[py::str(name)] = tensor_to_numpy(tensor);
                        }
                        callback_copy(result.request_id, result.success, outputs, result.error_message);
                    } catch (const py::error_already_set& e) {
                        // Log but don't propagate Python exceptions from callback
                    }
                });
        },
             py::arg("inputs"),
             py::arg("callback"),
             py::arg("output_names") = py::none(),
             "Submit async inference with callback.")

        .def("run_on_stream", [](AsyncSession& self, size_t stream_id,
                                const py::dict& inputs,
                                std::optional<std::vector<std::string>> output_names) {
            std::unordered_map<std::string, Tensor> feed;
            for (auto item : inputs) {
                std::string name = py::str(item.first);
                // Security: validate array conversion (HIGH-B3 fix)
                py::array arr = py::array::ensure(item.second);
                if (!arr) {
                    throw std::runtime_error(
                        "Input '" + name + "' could not be converted to numpy array");
                }
                feed[name] = tensor_from_numpy(arr);
            }
            return self.run_on_stream(stream_id,
                output_names.value_or(std::vector<std::string>{}), feed);
        },
             py::arg("stream_id"),
             py::arg("inputs"),
             py::arg("output_names") = py::none(),
             "Submit to specific stream.")

        .def("num_streams", &AsyncSession::num_streams,
             "Get number of streams")
        .def("stream", py::overload_cast<size_t>(&AsyncSession::stream),
             py::arg("id"),
             py::return_value_policy::reference,
             "Get stream context")
        .def("select_stream", &AsyncSession::select_stream,
             "Get least busy stream ID")
        .def("synchronize_all", &AsyncSession::synchronize_all,
             "Synchronize all streams")
        .def("synchronize", &AsyncSession::synchronize,
             py::arg("stream_id"),
             "Synchronize specific stream")
        .def("start", &AsyncSession::start,
             "Start async processing")
        .def("stop", &AsyncSession::stop,
             "Stop async processing")
        .def("is_running", &AsyncSession::is_running,
             "Check if running")
        .def("cancel", &AsyncSession::cancel,
             py::arg("request_id"),
             "Cancel pending request")
        .def("cancel_all", &AsyncSession::cancel_all,
             "Cancel all pending requests")
        .def("get_stats", &AsyncSession::get_stats,
             "Get statistics")
        .def("reset_stats", &AsyncSession::reset_stats,
             "Reset statistics")
        .def_property_readonly("options", &AsyncSession::options,
             "Get options");

    // ========================================================================
    // StreamingInference
    // ========================================================================
    py::class_<StreamingInference>(stream_mod, "StreamingInference",
        "Token-based streaming for sequence models.\n"
        "Useful for autoregressive inference.")
        .def(py::init<std::shared_ptr<InferenceSession>>(),
             py::arg("session"),
             "Create streaming inference.")

        .def("start", [](StreamingInference& self, const py::dict& initial_inputs) {
            std::unordered_map<std::string, Tensor> feed;
            for (auto item : initial_inputs) {
                std::string name = py::str(item.first);
                // Security: validate array conversion (CRIT-B3 fix)
                py::array arr = py::array::ensure(item.second);
                if (!arr) {
                    throw std::runtime_error(
                        "Input '" + name + "' could not be converted to numpy array");
                }
                feed[name] = tensor_from_numpy(arr);
            }
            self.start(feed);
        },
             py::arg("initial_inputs"),
             "Start streaming inference.")

        .def("step", [](StreamingInference& self, py::array input) {
            Tensor tensor = tensor_from_numpy(input);
            Tensor output = self.step(tensor);
            return tensor_to_numpy(output);
        },
             py::arg("input"),
             "Feed next input and get output.")

        .def("get_outputs", [](const StreamingInference& self) {
            py::list result;
            for (const auto& tensor : self.get_outputs()) {
                result.append(tensor_to_numpy(tensor));
            }
            return result;
        }, "Get accumulated outputs.")

        .def("is_complete", &StreamingInference::is_complete,
             "Check if streaming is complete")
        .def("stop", &StreamingInference::stop,
             "Stop streaming")
        .def("step_count", &StreamingInference::step_count,
             "Get number of steps taken")
        .def("set_step_callback", [](StreamingInference& self, py::function callback) {
            // Security: copy callback to ensure lifetime (HIGH-B4 fix)
            py::function callback_copy = callback;
            self.set_step_callback([callback_copy](const Tensor& output) {
                py::gil_scoped_acquire acquire;
                try {
                    callback_copy(tensor_to_numpy(output));
                } catch (const py::error_already_set& e) {
                    // Log but don't propagate Python exceptions from callback
                }
            });
        },
             py::arg("callback"),
             "Set callback for each step.");

    // ========================================================================
    // Factory functions
    // ========================================================================
    stream_mod.def("create_async_session",
        py::overload_cast<std::shared_ptr<InferenceSession>>(&create_async_session),
        py::arg("session"),
        "Create async session with default options.");

    stream_mod.def("create_async_session",
        py::overload_cast<std::shared_ptr<InferenceSession>, const AsyncOptions&>(
            &create_async_session),
        py::arg("session"),
        py::arg("options"),
        "Create async session with custom options.");
}
