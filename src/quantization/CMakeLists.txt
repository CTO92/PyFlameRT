# Quantization module sources
set(QUANTIZATION_SOURCES
    half_types.cpp
    quant_params.cpp
    quant_ops.cpp
    calibrator.cpp
    quantizer.cpp
)

# Add sources to parent scope
set(QUANTIZATION_SOURCES ${QUANTIZATION_SOURCES} PARENT_SCOPE)
