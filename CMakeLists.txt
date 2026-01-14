cmake_minimum_required(VERSION 3.18)
project(PyFlameRT
    VERSION 0.1.0
    LANGUAGES CXX
    DESCRIPTION "High-performance inference runtime for Cerebras WSE"
)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options
option(PYFLAME_RT_BUILD_TESTS "Build unit tests" ON)
option(PYFLAME_RT_BUILD_PYTHON "Build Python bindings" ON)
option(PYFLAME_RT_USE_EIGEN "Use Eigen for optimized operations" OFF)
option(PYFLAME_RT_BUILD_SHARED "Build shared library" ON)
option(PYFLAME_RT_BUILD_SERVING "Build serving infrastructure" ON)
option(PYFLAME_RT_USE_GRPC "Use gRPC for high-performance RPC" OFF)
option(PYFLAME_RT_USE_HTTPLIB "Use cpp-httplib for enhanced HTTP server" OFF)

# Output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Compiler warnings
if(MSVC)
    add_compile_options(/W4 /WX- /EHsc)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(Protobuf QUIET)
if(NOT Protobuf_FOUND)
    message(STATUS "Protobuf not found, model serialization will use simple binary format")
    set(PYFLAME_RT_USE_PROTOBUF OFF)
else()
    set(PYFLAME_RT_USE_PROTOBUF ON)
endif()

if(PYFLAME_RT_USE_EIGEN)
    find_package(Eigen3 3.3 REQUIRED NO_MODULE)
endif()

# Add subdirectories
add_subdirectory(third_party)
add_subdirectory(src)

if(PYFLAME_RT_BUILD_PYTHON)
    add_subdirectory(bindings)
endif()

if(PYFLAME_RT_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Install rules
include(GNUInstallDirs)
install(DIRECTORY include/pyflame_rt
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Export package configuration
install(TARGETS pyflame_rt_core
    EXPORT PyFlameRTTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(EXPORT PyFlameRTTargets
    FILE PyFlameRTTargets.cmake
    NAMESPACE PyFlameRT::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/PyFlameRT
)
