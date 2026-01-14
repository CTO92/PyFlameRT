# PyFlameRT Serving Infrastructure
# Phase 6: Model serving with HTTP/gRPC support

set(SERVING_SOURCES
    types.cpp
    metrics.cpp
    model_registry.cpp
    http_server.cpp
    model_server.cpp
)

# Serving static library
add_library(pyflame_rt_serving STATIC ${SERVING_SOURCES})

target_include_directories(pyflame_rt_serving
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link to core library
target_link_libraries(pyflame_rt_serving
    PUBLIC
        pyflame_rt_core
)

# Platform-specific dependencies
if(WIN32)
    target_link_libraries(pyflame_rt_serving PRIVATE ws2_32)
endif()

# Threading support
find_package(Threads REQUIRED)
target_link_libraries(pyflame_rt_serving PUBLIC Threads::Threads)

# Optional gRPC support
if(PYFLAME_RT_USE_GRPC)
    find_package(gRPC CONFIG REQUIRED)
    target_link_libraries(pyflame_rt_serving PUBLIC gRPC::grpc++)
    target_compile_definitions(pyflame_rt_serving PUBLIC PYFLAME_RT_USE_GRPC)

    # Generate gRPC service stubs
    # add_subdirectory(proto)
endif()

# Optional cpp-httplib for enhanced HTTP server
if(PYFLAME_RT_USE_HTTPLIB)
    target_compile_definitions(pyflame_rt_serving PUBLIC PYFLAME_RT_USE_HTTPLIB)
    # Note: cpp-httplib is header-only, include path should be set externally
endif()

# Install targets
install(TARGETS pyflame_rt_serving
    EXPORT PyFlameRTTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# Install headers
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/pyflame_rt/serving
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/pyflame_rt
    FILES_MATCHING PATTERN "*.hpp"
)
