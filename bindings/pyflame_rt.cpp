#include <pybind11/pybind11.h>

namespace py = pybind11;

// Forward declarations
void bind_types(py::module_& m);
void bind_tensor(py::module_& m);
void bind_session(py::module_& m);
void bind_import(py::module_& m);
void bind_import_convenience(py::module_& m);
void bind_opt(py::module_& m);
void bind_quantization(py::module_& m);

// Phase 5: Production features
void bind_cache(py::module_& m);
void bind_memory(py::module_& m);
void bind_batching(py::module_& m);
void bind_streaming(py::module_& m);

// Phase 6: Serving infrastructure
namespace pyflame_rt { namespace serving {
    void init_serving_bindings(py::module_& m);
}}

// Phase 7: Advanced optimization
void bind_pruning(py::module_& m);
void bind_distillation(py::module_& m);
void bind_custom_ops(py::module_& m);
void bind_partition(py::module_& m);

PYBIND11_MODULE(_pyflame_rt, m) {
    m.doc() = "PyFlameRT - High-performance inference runtime for Cerebras WSE";

    bind_types(m);
    bind_tensor(m);
    bind_session(m);
    bind_import(m);
    bind_import_convenience(m);
    bind_opt(m);
    bind_quantization(m);

    // Phase 5: Production features
    bind_cache(m);
    bind_memory(m);
    bind_batching(m);
    bind_streaming(m);

    // Phase 6: Serving infrastructure
#ifdef PYFLAME_RT_BUILD_SERVING
    pyflame_rt::serving::init_serving_bindings(m);
#endif

    // Phase 7: Advanced optimization
    bind_pruning(m);
    bind_distillation(m);
    bind_custom_ops(m);
    bind_partition(m);

    m.attr("__version__") = "0.1.0";
}
