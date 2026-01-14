# Python bindings using pybind11

set(BINDING_SOURCES
    pyflame_rt.cpp
    types_bind.cpp
    tensor_bind.cpp
    session_bind.cpp
    import_bind.cpp
    opt_bind.cpp
    quantization_bind.cpp
    # Phase 5: Production features
    cache_bind.cpp
    memory_bind.cpp
    batching_bind.cpp
    streaming_bind.cpp
    # Phase 7: Advanced optimization
    pruning_bind.cpp
    distillation_bind.cpp
    custom_op_bind.cpp
    partition_bind.cpp
)

# Phase 6: Serving bindings
if(PYFLAME_RT_BUILD_SERVING)
    list(APPEND BINDING_SOURCES serving_bind.cpp)
endif()

pybind11_add_module(_pyflame_rt MODULE ${BINDING_SOURCES})

# Link libraries
set(BINDING_LIBS
    pyflame_rt_core
    pyflame_rt_opt
    pyflame_rt_pruning
    pyflame_rt_distillation
    pyflame_rt_custom
    pyflame_rt_partition
)
if(PYFLAME_RT_BUILD_SERVING)
    list(APPEND BINDING_LIBS pyflame_rt_serving)
    target_compile_definitions(_pyflame_rt PRIVATE PYFLAME_RT_BUILD_SERVING)
endif()
target_link_libraries(_pyflame_rt PRIVATE ${BINDING_LIBS})

target_include_directories(_pyflame_rt PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

# Set output name without lib prefix
set_target_properties(_pyflame_rt PROPERTIES
    PREFIX ""
    OUTPUT_NAME "_pyflame_rt"
)

# Install to Python package location
install(TARGETS _pyflame_rt
    LIBRARY DESTINATION python/pyflame_rt
    RUNTIME DESTINATION python/pyflame_rt
)
