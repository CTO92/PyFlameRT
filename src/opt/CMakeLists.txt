# Phase 3: Graph Optimization

set(OPT_SOURCES
    pass_manager.cpp
    pattern_matcher.cpp
    constant_folding.cpp
    dead_code_elimination.cpp
    cse.cpp
    operator_fusion.cpp
    fusion_patterns.cpp
    layout_optimization.cpp
)

add_library(pyflame_rt_opt STATIC ${OPT_SOURCES})

target_include_directories(pyflame_rt_opt
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(pyflame_rt_opt
    PUBLIC
        pyflame_rt_core
)

# Install
install(TARGETS pyflame_rt_opt
    EXPORT PyFlameRTTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
