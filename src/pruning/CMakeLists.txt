# Pruning module

set(PRUNING_SOURCES
    pruning.cpp
    magnitude_pruning.cpp
    structured_pruning.cpp
    nm_sparsity.cpp
    sparse_tensor.cpp
)

add_library(pyflame_rt_pruning STATIC ${PRUNING_SOURCES})

target_include_directories(pyflame_rt_pruning PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(pyflame_rt_pruning PUBLIC
    pyflame_rt_core
)

set_target_properties(pyflame_rt_pruning PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
