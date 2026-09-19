cmake_minimum_required(VERSION 3.16)
project(glassjaw LANGUAGES C CXX)

add_library(gj_core STATIC
  src/dsp.cpp
  src/onnx/model.cpp
  src/onnx/tensor.cpp
  src/onnx/engine.cpp
)
target_include_directories(gj_core PUBLIC src)
target_compile_features(gj_core PUBLIC cxx_std_20)
target_compile_options(gj_core PRIVATE -Wall -Wextra -Werror)

add_executable(gj_tests
  tests/test_main.cpp
  tests/test_ring.cpp
  tests/test_detector.cpp
  tests/test_stats.cpp
  tests/test_dsp.cpp
  tests/test_onnx.cpp
)
target_link_libraries(gj_tests PRIVATE gj_core)
target_include_directories(gj_tests PRIVATE tests)
target_compile_options(gj_tests PRIVATE -Wall -Wextra -Werror)
target_compile_definitions(gj_tests PRIVATE GLASSJAW_MODELS_DIR="${CMAKE_SOURCE_DIR}/models")

enable_testing()
add_test(NAME unit COMMAND gj_tests)

add_subdirectory(sim)
