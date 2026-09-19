// onnx parser for the op subset the exported detector graph uses: see
// is_supported_op(). anything else is rejected at load time.
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "tensor.hpp"

namespace gj::onnx {

struct Attribute {
  enum class Kind { INT, INTS, FLOAT, FLOATS } kind;
  int64_t i = 0;
  std::vector<int64_t> ints;
  float f = 0.0f;
  std::vector<float> floats;
};

struct Node {
  std::string op;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  std::unordered_map<std::string, Attribute> attrs;
};

struct ValueInfo {
  std::string name;
  int64_t elem_type = 1;  // 1 = float
  std::vector<int64_t> dims;
};

struct Graph {
  std::string name;
  std::vector<Node> nodes;
  std::unordered_map<std::string, Tensor> initializers;
  std::vector<ValueInfo> inputs;          // all declared inputs
  std::vector<ValueInfo> inputs_runtime;  // inputs that are not initializers
  std::vector<ValueInfo> outputs;
};

struct Model {
  int64_t ir_version = 0;
  int64_t opset_version = 0;
  Graph graph;
};

// returns false and fills `error` on unsupported constructs.
bool parse_model(std::span<const std::byte> bytes, Model& out, std::string& error);


bool is_supported_op(const std::string& op);

}  // namespace gj::onnx
