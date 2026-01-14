# Partition module for PyFlameRT

add_library(pyflame_rt_partition STATIC
    cost_model.cpp
    graph_partitioner.cpp
    partitioned_executor.cpp
    wse_partition.cpp
)

target_link_libraries(pyflame_rt_partition
    PUBLIC pyflame_rt_core
)

target_include_directories(pyflame_rt_partition
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)

# Set C++17 standard
target_compile_features(pyflame_rt_partition PUBLIC cxx_std_17)
