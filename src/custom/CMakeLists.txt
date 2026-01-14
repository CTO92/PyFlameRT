# Custom operator module

set(CUSTOM_SOURCES
    custom_op.cpp
    custom_op_registry.cpp
)

add_library(pyflame_rt_custom STATIC ${CUSTOM_SOURCES})

target_include_directories(pyflame_rt_custom PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(pyflame_rt_custom PUBLIC
    pyflame_rt_core
)

set_target_properties(pyflame_rt_custom PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
