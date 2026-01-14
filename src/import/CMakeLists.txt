# PyFlameRT Import Module - CMake Configuration
# Model import functionality for ONNX, PyTorch, and TorchScript

set(IMPORT_SOURCES
    importer.cpp
    op_converter.cpp
    shape_inference.cpp
    onnx/onnx_parser.cpp
    onnx/onnx_importer.cpp
    onnx/onnx_op_converters.cpp
)

set(IMPORT_HEADERS
    ${PROJECT_SOURCE_DIR}/include/pyflame_rt/import/importer.hpp
    ${PROJECT_SOURCE_DIR}/include/pyflame_rt/import/op_converter.hpp
    ${PROJECT_SOURCE_DIR}/include/pyflame_rt/import/shape_inference.hpp
    ${PROJECT_SOURCE_DIR}/include/pyflame_rt/import/onnx_importer.hpp
)

# Create import library
add_library(pyflame_rt_import OBJECT ${IMPORT_SOURCES} ${IMPORT_HEADERS})

target_include_directories(pyflame_rt_import PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(pyflame_rt_import PUBLIC cxx_std_17)

# Add to parent scope for linking
set(IMPORT_OBJECTS $<TARGET_OBJECTS:pyflame_rt_import> PARENT_SCOPE)
