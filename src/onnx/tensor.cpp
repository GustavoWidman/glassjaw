#include "tensor.hpp"

namespace gj::onnx {

std::string Tensor::shape_str() const {
  std::string s = "[";
  for (std::size_t i = 0; i < dims.size(); ++i) {
    if (i) s += ",";
    s += std::to_string(dims[i]);
  }
  s += "]";
  return s;
}

}  // namespace gj::onnx
