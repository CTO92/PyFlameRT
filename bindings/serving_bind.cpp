#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>

#include "pyflame_rt/serving/server_config.hpp"
#include "pyflame_rt/serving/types.hpp"
#include "pyflame_rt/serving/model_registry.hpp"
#include "pyflame_rt/serving/model_server.hpp"
#include "pyflame_rt/serving/metrics.hpp"

namespace py = pybind11;

namespace pyflame_rt {
namespace serving {

void init_serving_bindings(py::module_& m) {
    auto serving = m.def_submodule("serving", "Model serving infrastructure");

    // ===========================================================================
    // Enums
    // ===========================================================================

    py::enum_<ServingErrorCode>(serving, "ServingErrorCode")
        .value("OK", ServingErrorCode::OK)
        .value("InvalidRequest", ServingErrorCode::InvalidRequest)
        .value("ModelNotFound", ServingErrorCode::ModelNotFound)
        .value("ModelNotReady", ServingErrorCode::ModelNotReady)
        .value("InferenceError", ServingErrorCode::InferenceError)
        .value("Timeout", ServingErrorCode::Timeout)
        .value("QueueFull", ServingErrorCode::QueueFull)
        .value("InternalError", ServingErrorCode::InternalError)
        .export_values();

    // ===========================================================================
    // Configuration Types
    // ===========================================================================

    py::class_<HTTPServerConfig>(serving, "HTTPServerConfig")
        .def(py::init<>())
        .def_readwrite("host", &HTTPServerConfig::host)
        .def_readwrite("port", &HTTPServerConfig::port)
        .def_readwrite("num_workers", &HTTPServerConfig::num_workers)
        .def_readwrite("max_request_size", &HTTPServerConfig::max_request_size)
        .def_readwrite("request_timeout_ms", &HTTPServerConfig::request_timeout_ms)
        .def_readwrite("keep_alive_timeout_ms", &HTTPServerConfig::keep_alive_timeout_ms)
        .def_readwrite("enable_cors", &HTTPServerConfig::enable_cors);

    py::class_<GRPCServerConfig>(serving, "GRPCServerConfig")
        .def(py::init<>())
        .def_readwrite("host", &GRPCServerConfig::host)
        .def_readwrite("port", &GRPCServerConfig::port)
        .def_readwrite("max_message_size", &GRPCServerConfig::max_message_size)
        .def_readwrite("num_completion_queues", &GRPCServerConfig::num_completion_queues);

    py::class_<TLSConfig>(serving, "TLSConfig")
        .def(py::init<>())
        .def_readwrite("enabled", &TLSConfig::enabled)
        .def_readwrite("cert_path", &TLSConfig::cert_path)
        .def_readwrite("key_path", &TLSConfig::key_path)
        .def_readwrite("ca_path", &TLSConfig::ca_path);

    py::class_<RateLimitConfig>(serving, "RateLimitConfig")
        .def(py::init<>())
        .def_readwrite("enabled", &RateLimitConfig::enabled)
        .def_readwrite("requests_per_second", &RateLimitConfig::requests_per_second)
        .def_readwrite("burst_size", &RateLimitConfig::burst_size);

    py::class_<ModelConfig>(serving, "ModelConfig")
        .def(py::init<>())
        .def_readwrite("name", &ModelConfig::name)
        .def_readwrite("model_path", &ModelConfig::model_path)
        .def_readwrite("version", &ModelConfig::version)
        .def_readwrite("max_batch_size", &ModelConfig::max_batch_size)
        .def_readwrite("preferred_batch_sizes", &ModelConfig::preferred_batch_sizes)
        .def_readwrite("batch_timeout_us", &ModelConfig::batch_timeout_us)
        .def_readwrite("enable_batching", &ModelConfig::enable_batching)
        .def_readwrite("warmup_requests", &ModelConfig::warmup_requests);

    py::class_<ServerConfig>(serving, "ServerConfig")
        .def(py::init<>())
        .def_readwrite("http", &ServerConfig::http)
        .def_readwrite("grpc", &ServerConfig::grpc)
        .def_readwrite("tls", &ServerConfig::tls)
        .def_readwrite("rate_limit", &ServerConfig::rate_limit)
        .def_readwrite("models", &ServerConfig::models)
        .def_readwrite("model_dir", &ServerConfig::model_dir)
        .def_readwrite("enable_metrics", &ServerConfig::enable_metrics)
        .def_readwrite("metrics_port", &ServerConfig::metrics_port)
        .def_readwrite("max_memory", &ServerConfig::max_memory);

    // ===========================================================================
    // Request/Response Types
    // ===========================================================================

    py::class_<InferRequest>(serving, "InferRequest")
        .def(py::init<>())
        .def_readwrite("request_id", &InferRequest::request_id)
        .def_readwrite("model_name", &InferRequest::model_name)
        .def_readwrite("model_version", &InferRequest::model_version)
        .def_readwrite("inputs", &InferRequest::inputs)
        .def_readwrite("output_names", &InferRequest::output_names)
        .def_readwrite("priority", &InferRequest::priority)
        .def_static("generate_id", &InferRequest::generate_id);

    py::class_<InferResponse>(serving, "InferResponse")
        .def(py::init<>())
        .def_readwrite("request_id", &InferResponse::request_id)
        .def_readwrite("model_name", &InferResponse::model_name)
        .def_readwrite("model_version", &InferResponse::model_version)
        .def_readwrite("outputs", &InferResponse::outputs)
        .def_readwrite("success", &InferResponse::success)
        .def_readwrite("error_code", &InferResponse::error_code)
        .def_readwrite("error_message", &InferResponse::error_message)
        .def_readwrite("latency_us", &InferResponse::latency_us)
        .def_readwrite("queue_time_us", &InferResponse::queue_time_us);

    // ===========================================================================
    // Model Statistics
    // ===========================================================================

    py::class_<ModelStats>(serving, "ModelStats")
        .def(py::init<>())
        .def_readwrite("total_requests", &ModelStats::total_requests)
        .def_readwrite("successful_requests", &ModelStats::successful_requests)
        .def_readwrite("failed_requests", &ModelStats::failed_requests)
        .def_readwrite("avg_latency_ms", &ModelStats::avg_latency_ms)
        .def_readwrite("p50_latency_ms", &ModelStats::p50_latency_ms)
        .def_readwrite("p95_latency_ms", &ModelStats::p95_latency_ms)
        .def_readwrite("p99_latency_ms", &ModelStats::p99_latency_ms);

    py::class_<IOSpec>(serving, "IOSpec")
        .def(py::init<>())
        .def_readwrite("name", &IOSpec::name)
        .def_readwrite("dtype", &IOSpec::dtype)
        .def_readwrite("shape", &IOSpec::shape);

    py::class_<ServingModelMetadata>(serving, "ServingModelMetadata")
        .def(py::init<>())
        .def_readwrite("name", &ServingModelMetadata::name)
        .def_readwrite("version", &ServingModelMetadata::version)
        .def_readwrite("platform", &ServingModelMetadata::platform)
        .def_readwrite("inputs", &ServingModelMetadata::inputs)
        .def_readwrite("outputs", &ServingModelMetadata::outputs);

    // ===========================================================================
    // Model Instance
    // ===========================================================================

    py::class_<ModelInstance, std::shared_ptr<ModelInstance>>(serving, "ModelInstance")
        .def("is_ready", &ModelInstance::is_ready)
        .def("infer", &ModelInstance::infer)
        .def("get_stats", &ModelInstance::get_stats)
        .def("reset_stats", &ModelInstance::reset_stats)
        .def("get_serving_metadata", &ModelInstance::get_serving_metadata)
        .def("input_names", &ModelInstance::input_names)
        .def("output_names", &ModelInstance::output_names)
        .def("config", &ModelInstance::config);

    // ===========================================================================
    // Model Registry
    // ===========================================================================

    py::class_<ModelVersionInfo>(serving, "ModelVersionInfo")
        .def(py::init<>())
        .def_readwrite("version", &ModelVersionInfo::version)
        .def_readwrite("path", &ModelVersionInfo::path)
        .def_readwrite("is_loaded", &ModelVersionInfo::is_loaded);

    py::class_<ModelRegistry::RegistryStats>(serving, "RegistryStats")
        .def(py::init<>())
        .def_readwrite("total_models", &ModelRegistry::RegistryStats::total_models)
        .def_readwrite("total_versions", &ModelRegistry::RegistryStats::total_versions)
        .def_readwrite("loaded_versions", &ModelRegistry::RegistryStats::loaded_versions)
        .def_readwrite("memory_used", &ModelRegistry::RegistryStats::memory_used);

    py::class_<ModelRegistry>(serving, "ModelRegistry")
        .def(py::init<size_t>(), py::arg("max_memory") = 0)
        .def("register_model", &ModelRegistry::register_model)
        .def("load_from_path", &ModelRegistry::load_from_path,
             py::arg("name"), py::arg("path"), py::arg("version") = "")
        .def("load_from_directory", &ModelRegistry::load_from_directory)
        .def("unload", &ModelRegistry::unload,
             py::arg("name"), py::arg("version") = "")
        .def("unload_all", &ModelRegistry::unload_all)
        .def("get", &ModelRegistry::get,
             py::arg("name"), py::arg("version") = "")
        .def("get_latest", &ModelRegistry::get_latest)
        .def("has", &ModelRegistry::has,
             py::arg("name"), py::arg("version") = "")
        .def("list_models", &ModelRegistry::list_models)
        .def("list_versions", &ModelRegistry::list_versions)
        .def("enable_hot_reload", &ModelRegistry::enable_hot_reload)
        .def("reload", &ModelRegistry::reload,
             py::arg("name"), py::arg("version") = "")
        .def("memory_used", &ModelRegistry::memory_used)
        .def("get_stats", &ModelRegistry::get_stats);

    // ===========================================================================
    // Model Server
    // ===========================================================================

    py::class_<ModelServer::ServerStats>(serving, "ServerStats")
        .def(py::init<>())
        .def_readwrite("total_requests", &ModelServer::ServerStats::total_requests)
        .def_readwrite("active_requests", &ModelServer::ServerStats::active_requests)
        .def_readwrite("total_models", &ModelServer::ServerStats::total_models)
        .def_readwrite("loaded_models", &ModelServer::ServerStats::loaded_models)
        .def_readwrite("uptime_seconds", &ModelServer::ServerStats::uptime_seconds)
        .def_readwrite("memory_used", &ModelServer::ServerStats::memory_used);

    py::class_<ModelServer>(serving, "ModelServer")
        .def(py::init<const ServerConfig&>())
        .def("start", &ModelServer::start,
             py::call_guard<py::gil_scoped_release>())
        .def("stop", &ModelServer::stop)
        .def("is_running", &ModelServer::is_running)
        .def("registry", py::overload_cast<>(&ModelServer::registry),
             py::return_value_policy::reference)
        .def("load_model", &ModelServer::load_model)
        .def("unload_model", &ModelServer::unload_model,
             py::arg("name"), py::arg("version") = "")
        .def("get_stats", &ModelServer::get_stats)
        .def("wait", &ModelServer::wait,
             py::call_guard<py::gil_scoped_release>())
        .def("http_port", &ModelServer::http_port)
        .def("on_ready", &ModelServer::on_ready)
        .def("on_error", &ModelServer::on_error);

    // ===========================================================================
    // Model Server Builder
    // ===========================================================================

    py::class_<ModelServerBuilder>(serving, "ModelServerBuilder")
        .def(py::init<>())
        .def("host", &ModelServerBuilder::host)
        .def("port", &ModelServerBuilder::port)
        .def("workers", &ModelServerBuilder::workers)
        .def("enable_metrics", &ModelServerBuilder::enable_metrics,
             py::arg("enable") = true)
        .def("metrics_port", &ModelServerBuilder::metrics_port)
        .def("add_model", py::overload_cast<const ModelConfig&>(
             &ModelServerBuilder::add_model))
        .def("add_model", py::overload_cast<const std::string&, const std::string&, const std::string&>(
             &ModelServerBuilder::add_model),
             py::arg("name"), py::arg("path"), py::arg("version") = "1")
        .def("model_dir", &ModelServerBuilder::model_dir)
        .def("enable_batching", &ModelServerBuilder::enable_batching,
             py::arg("max_batch_size") = 32, py::arg("timeout_us") = 5000)
        .def("request_timeout", &ModelServerBuilder::request_timeout)
        .def("build", &ModelServerBuilder::build);

    // ===========================================================================
    // Metrics
    // ===========================================================================

    auto metrics_mod = serving.def_submodule("metrics", "Prometheus metrics");

    py::class_<MetricsRegistry>(metrics_mod, "MetricsRegistry")
        .def_static("instance", &MetricsRegistry::instance,
                    py::return_value_policy::reference)
        .def("export_prometheus", &MetricsRegistry::export_prometheus)
        .def("reset", &MetricsRegistry::reset);

    // Convenience functions
    metrics_mod.def("request_total", &metrics::request_total);
    metrics_mod.def("request_latency", &metrics::request_latency);
    metrics_mod.def("inference_error", &metrics::inference_error);
    metrics_mod.def("model_loaded", &metrics::model_loaded);
    metrics_mod.def("batch_size", &metrics::batch_size);
    metrics_mod.def("queue_size", &metrics::queue_size);
    metrics_mod.def("request_active_inc", &metrics::request_active_inc);
    metrics_mod.def("request_active_dec", &metrics::request_active_dec);
}

} // namespace serving
} // namespace pyflame_rt
