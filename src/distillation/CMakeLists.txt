# Distillation module

set(DISTILLATION_SOURCES
    distillation_trainer.cpp
    distillation_losses.cpp
    student_architecture.cpp
    dataset.cpp
)

add_library(pyflame_rt_distillation STATIC ${DISTILLATION_SOURCES})

target_include_directories(pyflame_rt_distillation PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(pyflame_rt_distillation PUBLIC
    pyflame_rt_core
)

set_target_properties(pyflame_rt_distillation PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
