// executes a parsed graph node by node. onnx graphs are stored in
// topological order, so a single pass with a name->tensor map is enough.
#pragma once

#include <string>
#include <unordered_map>

#include "model.hpp"

namespace gj::onnx {

class Engine {
 public:
  // parse + validate; false with error message on anything unexpected.
  bool load(std::span<const std::byte> model_bytes, std::string& error);

  // run one inference. inputs must match the graph's runtime inputs (order
  // as declared). outputs land in `outputs` in declaration order.
  bool run(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
           std::string& error);

  // convenience for single-input/single-output models.
  bool run_f32(std::span<const float> input, std::span<float> output,
               std::string& error);

  std::size_t flops_hint() const { return flops_hint_; }

 private:
  Model model_;
  std::unordered_map<std::string, Tensor> buffers_;
  std::size_t flops_hint_ = 0;

  const Tensor* fetch(const std::string& name) const;
  Tensor* store(const std::string& name);

  bool exec_node(const Node& n, std::string& error);

  // op kernels
  bool op_conv(const Node& n, const Tensor& x, const Tensor& w, const Tensor* b,
               Tensor& y, std::string& error);
  bool op_qlinear_conv(const Node& n, std::string& error);
  bool op_matmul(const Tensor& a, const Tensor& b, Tensor& y);
  bool op_qlinear_matmul(const Node& n, std::string& error);
  bool op_maxpool(const Node& n, const Tensor* x, Tensor& y);
  bool op_quantize_linear(const Node& n, std::string& error);
  bool op_dequantize_linear(const Node& n, std::string& error);
};

}  // namespace gj::onnx
