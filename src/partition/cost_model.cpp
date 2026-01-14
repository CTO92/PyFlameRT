#include "pyflame_rt/partition/partition.hpp"
#include <cmath>
#include <algorithm>

namespace pyflame_rt {
namespace partition {

// ============================================================================
// CostModel Implementation
// ============================================================================

CostModel::CostModel() = default;
CostModel::~CostModel() = default;

double CostModel::estimate_compute_cost(
    const Node& node,
    const DeviceSpec& device) const {

    // Base cost estimation based on operation type
    double base_flops = 0.0;

    const std::string& op_type = node.op_type();

    // Get output shape to estimate computation
    size_t output_elements = 1;
    for (const auto& output : node.outputs()) {
        for (auto dim : output.shape) {
            if (dim.has_value() && dim.value() > 0) {
                output_elements *= static_cast<size_t>(dim.value());
            }
        }
    }

    // Estimate FLOPs based on operation type
    if (op_type == "MatMul" || op_type == "Gemm") {
        // MatMul: 2 * M * N * K FLOPs
        // Approximate using output size
        base_flops = 2.0 * static_cast<double>(output_elements);

        // Try to get actual dimensions from inputs
        if (node.inputs().size() >= 2) {
            const auto& a_shape = node.inputs()[0].shape;
            const auto& b_shape = node.inputs()[1].shape;
            if (a_shape.size() >= 2 && b_shape.size() >= 2) {
                int64_t k = 1;
                if (a_shape.back().has_value()) {
                    k = a_shape.back().value();
                }
                base_flops = 2.0 * static_cast<double>(output_elements) * static_cast<double>(k);
            }
        }
    } else if (op_type == "Conv") {
        // Conv: 2 * output_elements * kernel_size * in_channels
        base_flops = 2.0 * static_cast<double>(output_elements) * 9.0 * 64.0;  // Default estimate

        // Try to get kernel size from attributes
        auto attrs = node.attributes();
        if (attrs.count("kernel_shape")) {
            try {
                auto kernel_shape = std::any_cast<std::vector<int64_t>>(attrs.at("kernel_shape"));
                int64_t kernel_size = 1;
                for (auto k : kernel_shape) kernel_size *= k;
                base_flops = 2.0 * static_cast<double>(output_elements) * static_cast<double>(kernel_size);
            } catch (...) {
                // Use default estimate
            }
        }
    } else if (op_type == "BatchNormalization") {
        // BatchNorm: ~5 ops per element (mean, var, normalize, scale, shift)
        base_flops = 5.0 * static_cast<double>(output_elements);
    } else if (op_type == "Softmax") {
        // Softmax: ~3 ops per element (exp, sum, div)
        base_flops = 3.0 * static_cast<double>(output_elements);
    } else if (op_type == "Relu" || op_type == "LeakyRelu" || op_type == "Sigmoid" || op_type == "Tanh") {
        // Activation: 1 op per element
        base_flops = static_cast<double>(output_elements);
    } else if (op_type == "Add" || op_type == "Sub" || op_type == "Mul" || op_type == "Div") {
        // Element-wise: 1 op per element
        base_flops = static_cast<double>(output_elements);
    } else if (op_type == "ReduceSum" || op_type == "ReduceMean" || op_type == "ReduceMax") {
        // Reduction: roughly output_elements ops
        base_flops = static_cast<double>(output_elements);
    } else {
        // Default: assume 1 op per element
        base_flops = static_cast<double>(output_elements);
    }

    // Convert FLOPs to time based on device compute capacity
    double compute_time_us = 0.0;
    if (device.compute_flops > 0) {
        compute_time_us = (base_flops / device.compute_flops) * 1e6;  // Convert to microseconds
    } else {
        // Default: assume 1 GFLOP/s for CPU
        double default_flops = 1e9;
        if (device.type == DeviceType::WSE) {
            default_flops = 1e15;  // WSE: ~1 PFLOP/s
        } else if (device.type == DeviceType::GPU) {
            default_flops = 1e13;  // GPU: ~10 TFLOP/s
        }
        compute_time_us = (base_flops / default_flops) * 1e6;
    }

    return compute_time_us;
}

size_t CostModel::estimate_memory_cost(const Node& node) const {
    size_t total_bytes = 0;

    // Input memory
    for (const auto& input : node.inputs()) {
        size_t elements = 1;
        for (auto dim : input.shape) {
            if (dim.has_value() && dim.value() > 0) {
                elements *= static_cast<size_t>(dim.value());
            }
        }
        total_bytes += elements * dtype_size(input.dtype);
    }

    // Output memory
    for (const auto& output : node.outputs()) {
        size_t elements = 1;
        for (auto dim : output.shape) {
            if (dim.has_value() && dim.value() > 0) {
                elements *= static_cast<size_t>(dim.value());
            }
        }
        total_bytes += elements * dtype_size(output.dtype);
    }

    return total_bytes;
}

size_t CostModel::estimate_activation_memory(const Node& node) const {
    // Activation memory is output memory during forward pass
    size_t activation_bytes = 0;

    for (const auto& output : node.outputs()) {
        size_t elements = 1;
        for (auto dim : output.shape) {
            if (dim.has_value() && dim.value() > 0) {
                elements *= static_cast<size_t>(dim.value());
            }
        }
        activation_bytes += elements * dtype_size(output.dtype);
    }

    return activation_bytes;
}

double CostModel::estimate_comm_cost(
    size_t data_bytes,
    const DeviceSpec& src,
    const DeviceSpec& dst) const {

    // If same device, no communication cost
    if (src.device_id == dst.device_id && src.type == dst.type) {
        return 0.0;
    }

    // Communication time = latency + data_bytes / bandwidth
    double time_us = inter_device_latency_;
    time_us += (static_cast<double>(data_bytes) / inter_device_bandwidth_) * 1e6;

    return time_us;
}

double CostModel::estimate_allreduce_cost(
    size_t data_bytes,
    const std::vector<DeviceSpec>& devices) const {

    if (devices.size() <= 1) {
        return 0.0;
    }

    // Ring all-reduce: 2 * (n-1) / n * data_bytes total data transferred
    // Time = 2 * (n-1) * (latency + data_bytes / (n * bandwidth))
    size_t n = devices.size();
    double ring_factor = 2.0 * static_cast<double>(n - 1) / static_cast<double>(n);
    double data_per_step = static_cast<double>(data_bytes) / static_cast<double>(n);

    double time_us = static_cast<double>(2 * (n - 1)) * inter_device_latency_;
    time_us += ring_factor * (static_cast<double>(data_bytes) / inter_device_bandwidth_) * 1e6;

    return time_us;
}

void CostModel::set_inter_device_bandwidth(double bytes_per_sec) {
    inter_device_bandwidth_ = bytes_per_sec;
}

void CostModel::set_inter_device_latency(double microseconds) {
    inter_device_latency_ = microseconds;
}

} // namespace partition
} // namespace pyflame_rt
