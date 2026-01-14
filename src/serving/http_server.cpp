#include "pyflame_rt/serving/http_server.hpp"
#include "pyflame_rt/serving/model_registry.hpp"
#include "pyflame_rt/serving/metrics.hpp"
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <regex>
#include <cstring>

namespace pyflame_rt {
namespace serving {

// ============================================================================
// JSON Utilities Implementation
// ============================================================================

std::string dtype_to_json_string(DType dtype) {
    switch (dtype) {
        case DType::Float32: return "FP32";
        case DType::Float16: return "FP16";
        case DType::BFloat16: return "BF16";
        case DType::Float64: return "FP64";
        case DType::Int64: return "INT64";
        case DType::Int32: return "INT32";
        case DType::Int16: return "INT16";
        case DType::Int8: return "INT8";
        case DType::UInt8: return "UINT8";
        case DType::Bool: return "BOOL";
        default: return "FP32";
    }
}

DType dtype_from_json_string(const std::string& str) {
    if (str == "FP32" || str == "FLOAT" || str == "FLOAT32") return DType::Float32;
    if (str == "FP16" || str == "FLOAT16") return DType::Float16;
    if (str == "BF16" || str == "BFLOAT16") return DType::BFloat16;
    if (str == "FP64" || str == "DOUBLE" || str == "FLOAT64") return DType::Float64;
    if (str == "INT64") return DType::Int64;
    if (str == "INT32") return DType::Int32;
    if (str == "INT16") return DType::Int16;
    if (str == "INT8") return DType::Int8;
    if (str == "UINT8") return DType::UInt8;
    if (str == "BOOL") return DType::Bool;
    return DType::Float32;
}

std::string build_error_json(ServingErrorCode code, const std::string& message) {
    std::ostringstream oss;
    oss << "{\"error\":{\"code\":\"" << error_code_name(code)
        << "\",\"message\":\"" << message << "\"}}";
    return oss.str();
}

std::string build_model_list_json(const std::vector<ServingModelMetadata>& models) {
    std::ostringstream oss;
    oss << "{\"models\":[";

    bool first = true;
    for (const auto& model : models) {
        if (!first) oss << ",";
        first = false;

        oss << "{\"name\":\"" << model.name << "\""
            << ",\"version\":\"" << model.version << "\""
            << ",\"platform\":\"" << model.platform << "\""
            << ",\"state\":\"READY\"}";
    }

    oss << "]}";
    return oss.str();
}

std::string build_model_metadata_json(const ServingModelMetadata& metadata) {
    std::ostringstream oss;
    oss << "{\"name\":\"" << metadata.name << "\""
        << ",\"version\":\"" << metadata.version << "\""
        << ",\"platform\":\"" << metadata.platform << "\"";

    // Inputs
    oss << ",\"inputs\":[";
    bool first = true;
    for (const auto& input : metadata.inputs) {
        if (!first) oss << ",";
        first = false;

        oss << "{\"name\":\"" << input.name << "\""
            << ",\"datatype\":\"" << dtype_to_json_string(input.dtype) << "\""
            << ",\"shape\":[";

        bool first_dim = true;
        for (int64_t dim : input.shape) {
            if (!first_dim) oss << ",";
            first_dim = false;
            oss << dim;
        }
        oss << "]}";
    }
    oss << "]";

    // Outputs
    oss << ",\"outputs\":[";
    first = true;
    for (const auto& output : metadata.outputs) {
        if (!first) oss << ",";
        first = false;

        oss << "{\"name\":\"" << output.name << "\""
            << ",\"datatype\":\"" << dtype_to_json_string(output.dtype) << "\""
            << ",\"shape\":[";

        bool first_dim = true;
        for (int64_t dim : output.shape) {
            if (!first_dim) oss << ",";
            first_dim = false;
            oss << dim;
        }
        oss << "]}";
    }
    oss << "]";

    oss << "}";
    return oss.str();
}

std::string build_infer_response_json(const InferResponse& response) {
    std::ostringstream oss;
    oss << "{\"model_name\":\"" << response.model_name << "\""
        << ",\"model_version\":\"" << response.model_version << "\""
        << ",\"id\":\"" << response.request_id << "\"";

    if (!response.success) {
        oss << ",\"error\":{\"code\":\"" << error_code_name(response.error_code)
            << "\",\"message\":\"" << response.error_message << "\"}";
    }

    oss << ",\"outputs\":{";
    bool first = true;
    for (const auto& [name, tensor] : response.outputs) {
        if (!first) oss << ",";
        first = false;

        oss << "\"" << name << "\":{";
        oss << "\"shape\":[";
        bool first_dim = true;
        for (int64_t dim : tensor.shape()) {
            if (!first_dim) oss << ",";
            first_dim = false;
            oss << dim;
        }
        oss << "]";
        oss << ",\"datatype\":\"" << dtype_to_json_string(tensor.dtype()) << "\"";

        // Output data as array
        oss << ",\"data\":[";
        const float* data = static_cast<const float*>(tensor.data());
        int64_t num_elem = tensor.num_elements();
        for (int64_t i = 0; i < num_elem; ++i) {
            if (i > 0) oss << ",";
            oss << std::setprecision(8) << data[i];
        }
        oss << "]";
        oss << "}";
    }
    oss << "}";

    oss << "}";
    return oss.str();
}

// Simple JSON parser helpers
namespace {

std::string extract_json_string(const std::string& json, const std::string& key) {
    std::string pattern = "\"" + key + "\"\\s*:\\s*\"([^\"]+)\"";
    std::regex re(pattern);
    std::smatch match;
    if (std::regex_search(json, match, re)) {
        return match[1].str();
    }
    return "";
}

std::vector<int64_t> extract_json_int_array(const std::string& json, const std::string& key) {
    std::vector<int64_t> result;
    std::string pattern = "\"" + key + "\"\\s*:\\s*\\[([^\\]]+)\\]";
    std::regex re(pattern);
    std::smatch match;
    if (std::regex_search(json, match, re)) {
        std::string arr = match[1].str();
        std::regex num_re("-?\\d+");
        auto begin = std::sregex_iterator(arr.begin(), arr.end(), num_re);
        auto end = std::sregex_iterator();
        for (auto it = begin; it != end; ++it) {
            result.push_back(std::stoll(it->str()));
        }
    }
    return result;
}

std::vector<float> extract_json_float_array(const std::string& json, const std::string& key) {
    std::vector<float> result;
    std::string pattern = "\"" + key + "\"\\s*:\\s*\\[([^\\]]+)\\]";
    std::regex re(pattern);
    std::smatch match;
    if (std::regex_search(json, match, re)) {
        std::string arr = match[1].str();
        std::regex num_re("-?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?");
        auto begin = std::sregex_iterator(arr.begin(), arr.end(), num_re);
        auto end = std::sregex_iterator();
        for (auto it = begin; it != end; ++it) {
            result.push_back(std::stof(it->str()));
        }
    }
    return result;
}

std::string extract_json_object(const std::string& json, const std::string& key) {
    size_t key_pos = json.find("\"" + key + "\"");
    if (key_pos == std::string::npos) return "";

    size_t start = json.find('{', key_pos);
    if (start == std::string::npos) return "";

    int depth = 1;
    size_t end = start + 1;
    while (end < json.length() && depth > 0) {
        if (json[end] == '{') depth++;
        else if (json[end] == '}') depth--;
        end++;
    }

    return json.substr(start, end - start);
}

} // anonymous namespace

InferRequest parse_infer_request_json(const std::string& json,
                                       const std::string& model_name,
                                       const std::string& model_version) {
    InferRequest request;
    request.request_id = InferRequest::generate_id();
    request.model_name = model_name;
    request.model_version = model_version;
    request.arrival_time = std::chrono::steady_clock::now();

    // Parse inputs
    std::string inputs_obj = extract_json_object(json, "inputs");
    if (!inputs_obj.empty()) {
        // Find all input names
        std::regex name_re("\"([^\"]+)\"\\s*:\\s*\\{");
        auto begin = std::sregex_iterator(inputs_obj.begin(), inputs_obj.end(), name_re);
        auto end = std::sregex_iterator();

        for (auto it = begin; it != end; ++it) {
            std::string input_name = (*it)[1].str();
            std::string input_obj = extract_json_object(inputs_obj, input_name);

            if (!input_obj.empty()) {
                std::vector<int64_t> shape = extract_json_int_array(input_obj, "shape");
                std::string dtype_str = extract_json_string(input_obj, "datatype");
                std::vector<float> data = extract_json_float_array(input_obj, "data");

                if (!shape.empty() && !data.empty()) {
                    DType dtype = dtype_from_json_string(dtype_str);
                    Tensor tensor(shape, dtype);

                    // Copy data
                    float* tensor_data = static_cast<float*>(tensor.data());
                    size_t copy_size = std::min(data.size(),
                                                static_cast<size_t>(tensor.num_elements()));
                    std::copy(data.begin(), data.begin() + copy_size, tensor_data);

                    request.inputs[input_name] = std::move(tensor);
                }
            }
        }
    }

    // Parse output names if specified
    std::string outputs_pattern = "\"outputs\"\\s*:\\s*\\[([^\\]]+)\\]";
    std::regex outputs_re(outputs_pattern);
    std::smatch outputs_match;
    if (std::regex_search(json, outputs_match, outputs_re)) {
        std::string outputs_str = outputs_match[1].str();
        std::regex name_re("\"([^\"]+)\"");
        auto begin = std::sregex_iterator(outputs_str.begin(), outputs_str.end(), name_re);
        auto end = std::sregex_iterator();
        for (auto it = begin; it != end; ++it) {
            request.output_names.push_back((*it)[1].str());
        }
    }

    return request;
}

// ============================================================================
// HTTPServer Implementation
// ============================================================================

HTTPServer::HTTPServer(const HTTPServerConfig& config)
    : config_(config)
{
}

HTTPServer::~HTTPServer() = default;

void HTTPServer::route(const std::string& method, const std::string& path,
                       RouteHandler handler) {
    routes_.emplace_back(method, path, std::move(handler));
}

bool HTTPServer::match_path(const std::string& pattern, const std::string& path,
                            std::unordered_map<std::string, std::string>& params) const {
    // Convert pattern to regex, replacing :param with named capture
    std::string regex_pattern = "^";
    std::vector<std::string> param_names;

    size_t i = 0;
    while (i < pattern.length()) {
        if (pattern[i] == ':') {
            // Parameter
            size_t start = i + 1;
            while (i + 1 < pattern.length() && pattern[i + 1] != '/') {
                i++;
            }
            std::string param_name = pattern.substr(start, i - start + 1);
            param_names.push_back(param_name);
            regex_pattern += "([^/]+)";
        } else {
            // Escape special regex characters
            if (pattern[i] == '.' || pattern[i] == '?' || pattern[i] == '+' ||
                pattern[i] == '*' || pattern[i] == '(' || pattern[i] == ')' ||
                pattern[i] == '[' || pattern[i] == ']' || pattern[i] == '{' ||
                pattern[i] == '}' || pattern[i] == '\\' || pattern[i] == '^' ||
                pattern[i] == '$' || pattern[i] == '|') {
                regex_pattern += '\\';
            }
            regex_pattern += pattern[i];
        }
        i++;
    }
    regex_pattern += "$";

    std::regex re(regex_pattern);
    std::smatch match;
    if (std::regex_match(path, match, re)) {
        for (size_t j = 0; j < param_names.size() && j + 1 < match.size(); ++j) {
            params[param_names[j]] = match[j + 1].str();
        }
        return true;
    }
    return false;
}

HTTPResponse HTTPServer::handle_request(const HTTPRequest& request) {
    // Find matching route
    for (const auto& [method, pattern, handler] : routes_) {
        if (request.method != method) continue;

        std::unordered_map<std::string, std::string> params;
        if (match_path(config_.base_path + pattern, request.path, params)) {
            HTTPRequest req = request;
            req.path_params = params;
            return handler(req);
        }
    }

    // No route matched
    HTTPResponse response;
    response.set_error(404, "Not Found");
    return response;
}

void HTTPServer::setup_inference_routes() {
    // POST /models/:model/infer
    route("POST", "/models/:model/infer", [this](const HTTPRequest& req) -> HTTPResponse {
        HTTPResponse response;

        if (!registry_) {
            response.set_error(500, "Model registry not configured");
            return response;
        }

        std::string model_name = req.path_params.at("model");
        std::string model_version;

        // Check if version is in path params
        auto ver_it = req.path_params.find("version");
        if (ver_it != req.path_params.end()) {
            model_version = ver_it->second;
        }

        // Get model
        auto instance = registry_->get(model_name, model_version);
        if (!instance) {
            response.set_json(build_error_json(ServingErrorCode::ModelNotFound,
                                               "Model not found: " + model_name), 404);
            return response;
        }

        // Parse request
        try {
            InferRequest infer_req = parse_infer_request_json(
                req.body, model_name, instance->version());

            // Run inference
            InferResponse infer_res = instance->infer(infer_req);

            if (infer_res.success) {
                response.set_json(build_infer_response_json(infer_res), 200);
            } else {
                response.set_json(build_error_json(infer_res.error_code,
                                                   infer_res.error_message), 500);
            }

        } catch (const std::exception& e) {
            response.set_json(build_error_json(ServingErrorCode::InvalidRequest,
                                               e.what()), 400);
        }

        return response;
    });

    // POST /models/:model/versions/:version/infer
    route("POST", "/models/:model/versions/:version/infer",
          [this](const HTTPRequest& req) -> HTTPResponse {
        HTTPResponse response;

        if (!registry_) {
            response.set_error(500, "Model registry not configured");
            return response;
        }

        std::string model_name = req.path_params.at("model");
        std::string model_version = req.path_params.at("version");

        auto instance = registry_->get(model_name, model_version);
        if (!instance) {
            response.set_json(build_error_json(ServingErrorCode::ModelNotFound,
                "Model not found: " + model_name + " v" + model_version), 404);
            return response;
        }

        try {
            InferRequest infer_req = parse_infer_request_json(
                req.body, model_name, model_version);
            InferResponse infer_res = instance->infer(infer_req);

            if (infer_res.success) {
                response.set_json(build_infer_response_json(infer_res), 200);
            } else {
                response.set_json(build_error_json(infer_res.error_code,
                                                   infer_res.error_message), 500);
            }
        } catch (const std::exception& e) {
            response.set_json(build_error_json(ServingErrorCode::InvalidRequest,
                                               e.what()), 400);
        }

        return response;
    });

    // GET /models
    route("GET", "/models", [this](const HTTPRequest&) -> HTTPResponse {
        HTTPResponse response;

        if (!registry_) {
            response.set_error(500, "Model registry not configured");
            return response;
        }

        std::vector<ServingModelMetadata> models;
        for (const auto& name : registry_->list_models()) {
            auto instance = registry_->get_latest(name);
            if (instance) {
                models.push_back(instance->get_serving_metadata());
            }
        }

        response.set_json(build_model_list_json(models), 200);
        return response;
    });

    // GET /models/:model
    route("GET", "/models/:model", [this](const HTTPRequest& req) -> HTTPResponse {
        HTTPResponse response;

        if (!registry_) {
            response.set_error(500, "Model registry not configured");
            return response;
        }

        std::string model_name = req.path_params.at("model");
        auto instance = registry_->get_latest(model_name);

        if (!instance) {
            response.set_json(build_error_json(ServingErrorCode::ModelNotFound,
                                               "Model not found: " + model_name), 404);
            return response;
        }

        response.set_json(build_model_metadata_json(instance->get_serving_metadata()), 200);
        return response;
    });

    // GET /health/live
    route("GET", "/health/live", [](const HTTPRequest&) -> HTTPResponse {
        HTTPResponse response;
        response.set_json("{\"status\":\"live\"}", 200);
        return response;
    });

    // GET /health/ready
    route("GET", "/health/ready", [this](const HTTPRequest&) -> HTTPResponse {
        HTTPResponse response;

        if (!registry_) {
            response.set_json("{\"status\":\"not ready\",\"reason\":\"no registry\"}", 503);
            return response;
        }

        auto models = registry_->list_models();
        if (models.empty()) {
            response.set_json("{\"status\":\"not ready\",\"reason\":\"no models\"}", 503);
            return response;
        }

        // Check if at least one model is ready
        bool any_ready = false;
        for (const auto& name : models) {
            auto instance = registry_->get_latest(name);
            if (instance && instance->is_ready()) {
                any_ready = true;
                break;
            }
        }

        if (any_ready) {
            response.set_json("{\"status\":\"ready\"}", 200);
        } else {
            response.set_json("{\"status\":\"not ready\",\"reason\":\"models loading\"}", 503);
        }

        return response;
    });
}

// ============================================================================
// Simple HTTP Server Implementation (for platforms without cpp-httplib)
// ============================================================================

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <arpa/inet.h>
#endif

class SimpleHTTPServer : public HTTPServer {
public:
    SimpleHTTPServer(const HTTPServerConfig& config)
        : HTTPServer(config)
    {
        setup_inference_routes();
    }

    ~SimpleHTTPServer() override {
        stop();
    }

    void start() override {
        if (running_) return;

#ifdef _WIN32
        WSADATA wsa_data;
        WSAStartup(MAKEWORD(2, 2), &wsa_data);
#endif

        server_socket_ = socket(AF_INET, SOCK_STREAM, 0);
        if (server_socket_ < 0) {
            throw std::runtime_error("Failed to create socket");
        }

        // Enable address reuse
        int opt = 1;
#ifdef _WIN32
        setsockopt(server_socket_, SOL_SOCKET, SO_REUSEADDR,
                   reinterpret_cast<const char*>(&opt), sizeof(opt));
#else
        setsockopt(server_socket_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#endif

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = htons(config_.port);

        if (bind(server_socket_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
            close_socket(server_socket_);
            throw std::runtime_error("Failed to bind to port " + std::to_string(config_.port));
        }

        if (listen(server_socket_, 128) < 0) {
            close_socket(server_socket_);
            throw std::runtime_error("Failed to listen on socket");
        }

        running_ = true;

        // Start worker threads
        size_t num_workers = config_.num_workers;
        if (num_workers == 0) {
            num_workers = std::thread::hardware_concurrency();
            if (num_workers == 0) num_workers = 4;
        }

        for (size_t i = 0; i < num_workers; ++i) {
            workers_.emplace_back(&SimpleHTTPServer::worker_loop, this);
        }
    }

    void stop() override {
        if (!running_) return;

        running_ = false;
        close_socket(server_socket_);

        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        workers_.clear();

#ifdef _WIN32
        WSACleanup();
#endif
    }

    bool is_running() const override {
        return running_;
    }

    void wait() override {
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

private:
#ifdef _WIN32
    using socket_t = SOCKET;
    static constexpr socket_t INVALID_SOCK = INVALID_SOCKET;
#else
    using socket_t = int;
    static constexpr socket_t INVALID_SOCK = -1;
#endif

    socket_t server_socket_ = INVALID_SOCK;
    std::atomic<bool> running_{false};
    std::vector<std::thread> workers_;

    void close_socket(socket_t sock) {
        if (sock != INVALID_SOCK) {
#ifdef _WIN32
            closesocket(sock);
#else
            close(sock);
#endif
        }
    }

    void worker_loop() {
        while (running_) {
            sockaddr_in client_addr{};
            socklen_t client_len = sizeof(client_addr);

            socket_t client_socket = accept(server_socket_,
                reinterpret_cast<sockaddr*>(&client_addr), &client_len);

            if (client_socket == INVALID_SOCK) {
                if (running_) continue;
                break;
            }

            handle_connection(client_socket, client_addr);
            close_socket(client_socket);
        }
    }

    void handle_connection(socket_t client_socket, const sockaddr_in& client_addr) {
        // Read request
        std::string request_data;
        char buffer[4096];
        ssize_t bytes_read;

        while ((bytes_read = recv(client_socket, buffer, sizeof(buffer) - 1, 0)) > 0) {
            buffer[bytes_read] = '\0';
            request_data += buffer;

            // Check if we have complete headers
            if (request_data.find("\r\n\r\n") != std::string::npos) {
                // Check for Content-Length and read body if needed
                size_t content_length = 0;
                std::string cl_header = "Content-Length: ";
                size_t cl_pos = request_data.find(cl_header);
                if (cl_pos != std::string::npos) {
                    size_t cl_end = request_data.find("\r\n", cl_pos);
                    std::string cl_value = request_data.substr(
                        cl_pos + cl_header.length(),
                        cl_end - cl_pos - cl_header.length());
                    content_length = std::stoull(cl_value);
                }

                size_t header_end = request_data.find("\r\n\r\n") + 4;
                size_t body_received = request_data.length() - header_end;

                // Read remaining body if needed
                while (body_received < content_length) {
                    bytes_read = recv(client_socket, buffer, sizeof(buffer) - 1, 0);
                    if (bytes_read <= 0) break;
                    buffer[bytes_read] = '\0';
                    request_data += buffer;
                    body_received += bytes_read;
                }
                break;
            }
        }

        if (request_data.empty()) return;

        // Parse HTTP request
        HTTPRequest request = parse_http_request(request_data);
        request.client_ip = inet_ntoa(client_addr.sin_addr);

        // Handle request
        HTTPResponse response = handle_request(request);

        // Send response
        std::string response_data = build_http_response(response);
        send(client_socket, response_data.c_str(), response_data.length(), 0);
    }

    HTTPRequest parse_http_request(const std::string& data) {
        HTTPRequest request;

        // Parse request line
        size_t line_end = data.find("\r\n");
        std::string request_line = data.substr(0, line_end);

        size_t method_end = request_line.find(' ');
        request.method = request_line.substr(0, method_end);

        size_t path_end = request_line.find(' ', method_end + 1);
        std::string path_with_query = request_line.substr(method_end + 1,
                                                          path_end - method_end - 1);

        // Split path and query
        size_t query_start = path_with_query.find('?');
        if (query_start != std::string::npos) {
            request.path = path_with_query.substr(0, query_start);
            std::string query = path_with_query.substr(query_start + 1);
            // Parse query params
            size_t pos = 0;
            while (pos < query.length()) {
                size_t eq = query.find('=', pos);
                size_t amp = query.find('&', pos);
                if (eq == std::string::npos) break;
                std::string key = query.substr(pos, eq - pos);
                std::string value = query.substr(eq + 1,
                    (amp == std::string::npos ? query.length() : amp) - eq - 1);
                request.query_params[key] = value;
                pos = (amp == std::string::npos) ? query.length() : amp + 1;
            }
        } else {
            request.path = path_with_query;
        }

        // Parse headers
        size_t header_start = line_end + 2;
        size_t header_end = data.find("\r\n\r\n");

        while (header_start < header_end) {
            size_t next_line = data.find("\r\n", header_start);
            std::string header_line = data.substr(header_start, next_line - header_start);

            size_t colon = header_line.find(':');
            if (colon != std::string::npos) {
                std::string key = header_line.substr(0, colon);
                std::string value = header_line.substr(colon + 1);
                // Trim whitespace
                while (!value.empty() && value[0] == ' ') value = value.substr(1);
                request.headers[key] = value;
            }

            header_start = next_line + 2;
        }

        // Body
        if (header_end + 4 < data.length()) {
            request.body = data.substr(header_end + 4);
        }

        return request;
    }

    std::string build_http_response(const HTTPResponse& response) {
        std::ostringstream oss;

        // Status line
        oss << "HTTP/1.1 " << response.status_code << " ";
        switch (response.status_code) {
            case 200: oss << "OK"; break;
            case 201: oss << "Created"; break;
            case 204: oss << "No Content"; break;
            case 400: oss << "Bad Request"; break;
            case 404: oss << "Not Found"; break;
            case 500: oss << "Internal Server Error"; break;
            case 503: oss << "Service Unavailable"; break;
            default: oss << "Unknown"; break;
        }
        oss << "\r\n";

        // Headers
        oss << "Content-Type: " << response.content_type << "\r\n";
        oss << "Content-Length: " << response.body.length() << "\r\n";

        if (config_.enable_cors) {
            oss << "Access-Control-Allow-Origin: *\r\n";
            oss << "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n";
            oss << "Access-Control-Allow-Headers: Content-Type\r\n";
        }

        for (const auto& [key, value] : response.headers) {
            oss << key << ": " << value << "\r\n";
        }

        oss << "\r\n";
        oss << response.body;

        return oss.str();
    }
};

std::unique_ptr<HTTPServer> create_http_server(const HTTPServerConfig& config) {
    return std::make_unique<SimpleHTTPServer>(config);
}

} // namespace serving
} // namespace pyflame_rt
