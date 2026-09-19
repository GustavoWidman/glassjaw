// glassjaw — value tensors for the mini onnx engine.
#pragma once

#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <variant>
#include <vector>

#include "../pipeline.hpp"

namespace gj::onnx {

struct Tensor {
  enum class Kind { F32, INT8, INT32, INT64 };
  Kind kind = Kind::F32;
  bool u8 = false;              // kind INT8 holding uint8 bit patterns
  std::vector<int64_t> dims;
  std::vector<float> f32;
  std::vector<int8_t> i8;
  std::vector<int32_t> i32;
  std::vector<int64_t> i64;

  int64_t numel() const {
    int64_t n = 1;
    for (int64_t d : dims) n *= d;
    return n;
  }
  std::string shape_str() const;
  // NCHW accessors for conv-style tensors
  int64_t n() const { return dims.size() > 0 ? dims[0] : 1; }
  int64_t c() const { return dims.size() > 1 ? dims[1] : 1; }
  int64_t h() const { return dims.size() > 2 ? dims[2] : 1; }
  int64_t w() const { return dims.size() > 3 ? dims[3] : 1; }
};

}  // namespace gj::onnx
