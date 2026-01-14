# PyFlameRT Core Library

# Phase 3: Graph Optimization
add_subdirectory(opt)

# Phase 4: Quantization
add_subdirectory(quantization)

# Phase 6: Serving Infrastructure
if(PYFLAME_RT_BUILD_SERVING)
    add_subdirectory(serving)
endif()

# Phase 7: Advanced Optimization
add_subdirectory(pruning)
add_subdirectory(distillation)
add_subdirectory(custom)
add_subdirectory(partition)

set(PYFLAME_RT_SOURCES
    tensor.cpp
    node.cpp
    graph.cpp
    registry.cpp
    session.cpp
    options.cpp
    backends/cpu/executor.cpp
    backends/cpu/ops/math.cpp
    backends/cpu/ops/activation.cpp
    backends/cpu/ops/tensor_ops.cpp
    backends/cpu/ops/reduction.cpp
    backends/cpu/ops/nn.cpp
    io/loader.cpp
    io/pyflame_format.cpp
    # Phase 2: Import module
    import/importer.cpp
    import/op_converter.cpp
    import/shape_inference.cpp
    # ONNX importer
    import/onnx/onnx_parser.cpp
    import/onnx/onnx_importer.cpp
    import/onnx/onnx_op_converters.cpp
    # PyTorch importer
    import/pytorch/pickle_parser.cpp
    import/pytorch/pytorch_importer.cpp
    # TorchScript importer
    import/torchscript/torchscript_importer.cpp
    # Phase 4: Quantization
    quantization/half_types.cpp
    quantization/quant_params.cpp
    quantization/quant_ops.cpp
    quantization/calibrator.cpp
    quantization/quantizer.cpp
    # Quantized CPU operators
    backends/cpu/ops/quantized_ops.cpp
    # Phase 5: Production features
    memory/memory_pool.cpp
    cache/binary_cache.cpp
    cache/mmap_loader.cpp
    batching/dynamic_batcher.cpp
    streaming/async_session.cpp
)

# Core static library
add_library(pyflame_rt_core STATIC ${PYFLAME_RT_SOURCES})

target_include_directories(pyflame_rt_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

if(PYFLAME_RT_USE_PROTOBUF)
    target_link_libraries(pyflame_rt_core PUBLIC protobuf::libprotobuf)
    target_compile_definitions(pyflame_rt_core PUBLIC PYFLAME_RT_USE_PROTOBUF)
endif()

if(PYFLAME_RT_USE_EIGEN)
    target_link_libraries(pyflame_rt_core PUBLIC Eigen3::Eigen)
    target_compile_definitions(pyflame_rt_core PUBLIC PYFLAME_RT_USE_EIGEN)
endif()

# Shared library (optional)
if(PYFLAME_RT_BUILD_SHARED)
    add_library(pyflame_rt SHARED ${PYFLAME_RT_SOURCES})

    target_include_directories(pyflame_rt
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
    )

    if(PYFLAME_RT_USE_PROTOBUF)
        target_link_libraries(pyflame_rt PUBLIC protobuf::libprotobuf)
        target_compile_definitions(pyflame_rt PUBLIC PYFLAME_RT_USE_PROTOBUF)
    endif()

    if(PYFLAME_RT_USE_EIGEN)
        target_link_libraries(pyflame_rt PUBLIC Eigen3::Eigen)
        target_compile_definitions(pyflame_rt PUBLIC PYFLAME_RT_USE_EIGEN)
    endif()

    set_target_properties(pyflame_rt PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
    )

    install(TARGETS pyflame_rt
        EXPORT PyFlameRTTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()
