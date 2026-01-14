# C++ unit tests

set(TEST_SOURCES
    test_types.cpp
    test_tensor.cpp
    test_graph.cpp
    test_registry.cpp
    test_executor.cpp
    test_options.cpp
    test_loader.cpp
    # Phase 7: Advanced optimization tests
    test_pruning.cpp
    test_distillation.cpp
    test_custom_op.cpp
    test_partition.cpp
)

# Create test executable
add_executable(pyflame_rt_tests ${TEST_SOURCES})

target_link_libraries(pyflame_rt_tests
    PRIVATE
        pyflame_rt_core
        pyflame_rt_pruning
        pyflame_rt_distillation
        pyflame_rt_custom
        pyflame_rt_partition
        GTest::gtest
        GTest::gtest_main
)

target_include_directories(pyflame_rt_tests
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
)

# Register with CTest
include(GoogleTest)
gtest_discover_tests(pyflame_rt_tests)
