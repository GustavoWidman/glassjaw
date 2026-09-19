idf_component_register(
  SRCS "main.cc" "../../src/app/pipeline.cpp" "../../src/dsp.cpp"
       "../../src/onnx/model.cpp" "../../src/onnx/tensor.cpp" "../../src/onnx/engine.cpp"
  INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/../../src" "${CMAKE_CURRENT_LIST_DIR}"
  REQUIRES driver esp_timer
  EMBED_FILES "model/detector_int8.onnx"
)
target_compile_options(${COMPONENT_LIB} PRIVATE -Wall -Wextra)
