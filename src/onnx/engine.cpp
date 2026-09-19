#include "engine.hpp"

#include <algorithm>
#include <cmath>
#if defined(ESP_PLATFORM)
#include "esp_heap_caps.h"
#include "esp_log.h"
#endif
#include <cstdlib>
#include <limits>

namespace gj::onnx {

namespace {

// zero points may live in uint8 or int8 tensors; widen with the right sign
inline int32_t zp_of(const Tensor* t) {
  if (!t || t->i8.empty()) return 0;
  return t->u8 ? static_cast<int32_t>(static_cast<uint8_t>(t->i8[0]))
               : static_cast<int32_t>(t->i8[0]);
}
inline int32_t val_of(const Tensor* t, std::size_t i) {
  int8_t v = t->i8[i];
  return t->u8 ? static_cast<int32_t>(static_cast<uint8_t>(v)) : static_cast<int32_t>(v);
}

int64_t attr_int(const Node& n, const std::string& key, int64_t dflt) {
  auto it = n.attrs.find(key);
  return it == n.attrs.end() ? dflt : it->second.i;
}

std::vector<int64_t> attr_ints(const Node& n, const std::string& key) {
  auto it = n.attrs.find(key);
  return it == n.attrs.end() ? std::vector<int64_t>{} : it->second.ints;
}

float attr_f(const Node& n, const std::string& key, float dflt) {
  auto it = n.attrs.find(key);
  return it == n.attrs.end() ? dflt : it->second.f;
}

Tensor make_f32(std::vector<int64_t> dims) {
  Tensor t;
  t.kind = Tensor::Kind::F32;
  t.dims = std::move(dims);
  int64_t n = t.numel();
  t.f32.assign(static_cast<std::size_t>(n), 0.0f);
  return t;
}

}  // namespace

bool Engine::load(std::span<const std::byte> bytes, std::string& error) {
  if (!parse_model(bytes, model_, error)) return false;
  buffers_.clear();
  for (auto& [name, t] : model_.graph.initializers) {
    if (t.numel() == 0 && t.dims.empty()) {
      error = "initializer " + name + " has unsupported dtype or empty data";
      return false;
    }
  }
  return true;
}

const Tensor* Engine::fetch(const std::string& name) const {
  auto it = buffers_.find(name);
  if (it != buffers_.end()) return &it->second;
  auto init = model_.graph.initializers.find(name);
  return init == model_.graph.initializers.end() ? nullptr : &init->second;
}

Tensor* Engine::store(const std::string& name) { return &buffers_[name]; }

bool Engine::run(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                 std::string& error) {
  buffers_.clear();
  if (inputs.size() != model_.graph.inputs_runtime.size()) {
    error = "expected " + std::to_string(model_.graph.inputs_runtime.size()) +
            " inputs, got " + std::to_string(inputs.size());
    return false;
  }
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    buffers_.emplace(model_.graph.inputs_runtime[i].name, inputs[i]);
  }

  // liveness: tensors die once every consumer has executed. without this the
  // qdq graph holds ~700 kb of float intermediates alive on a part with
  // ~300 kb of heap and the first inference throws bad_alloc.
  std::unordered_map<std::string, int> pending;
  for (const Node& n : model_.graph.nodes) {
    for (const std::string& in : n.inputs) {
      if (!in.empty()) ++pending[in];
    }
  }
  auto is_output = [this](const std::string& name) {
    for (auto& vi : model_.graph.outputs) {
      if (vi.name == name) return true;
    }
    return false;
  };

  for (const Node& n : model_.graph.nodes) {
    if (!exec_node(n, error)) return false;
    for (const std::string& in : n.inputs) {
      if (in.empty()) continue;
      auto it = pending.find(in);
      if (it != pending.end() && --it->second <= 0 && !is_output(in)) {
        buffers_.erase(in);
      }
    }
  }
  outputs.clear();
  for (auto& vi : model_.graph.outputs) {
    const Tensor* t = fetch(vi.name);
    if (!t) {
      error = "graph output " + vi.name + " was never produced";
      return false;
    }
    outputs.push_back(*t);
  }
  return true;
}

bool Engine::run_f32(std::span<const float> input, std::span<float> output,
                     std::string& error) {
  if (model_.graph.inputs_runtime.size() != 1 || model_.graph.outputs.size() != 1) {
    error = "run_f32 expects a 1-in/1-out graph";
    return false;
  }
  Tensor in;
  in.kind = Tensor::Kind::F32;
  in.dims = model_.graph.inputs_runtime[0].dims;
  in.f32.assign(input.begin(), input.end());
  std::vector<Tensor> outs;
  if (!run({in}, outs, error)) return false;
  if (outs[0].kind != Tensor::Kind::F32 || outs[0].f32.size() != output.size()) {
    error = "output shape mismatch";
    return false;
  }
  std::copy(outs[0].f32.begin(), outs[0].f32.end(), output.begin());
  return true;
}

// ---- kernels -------------------------------------------------------------

bool Engine::op_conv(const Node& n, const Tensor& x, const Tensor& w, const Tensor* b,
                     Tensor& y, std::string& error) {
  if (x.dims.size() != 4 || w.dims.size() != 4) {
    error = "conv expects 4d nhwc/nchw operands";
    return false;
  }
  auto pads = attr_ints(n, "pads");           // [t, l, b, r]
  auto strides = attr_ints(n, "strides");
  auto dilations = attr_ints(n, "dilations");
  if (pads.size() != 4) pads = {0, 0, 0, 0};
  if (strides.size() != 2) strides = {1, 1};
  if (dilations.size() != 2) dilations = {1, 1};
  int64_t group = attr_int(n, "group", 1);
  if (group != 1) {
    error = "grouped conv unsupported";
    return false;
  }

  const int64_t N = x.n(), C = x.c(), H = x.h(), W = x.w();
  const int64_t M = w.dims[0], Cw = w.dims[1], kH = w.dims[2], kW = w.dims[3];
  if (Cw != C) {
    error = "conv channel mismatch";
    return false;
  }
  const int64_t pH = pads[0], pW = pads[1];
  const int64_t sH = strides[0], sW = strides[1];
  const int64_t dH = dilations[0], dW = dilations[1];
  const int64_t oH = (H + pH + pads[2] - ((kH - 1) * dH + 1)) / sH + 1;
  const int64_t oW = (W + pW + pads[3] - ((kW - 1) * dW + 1)) / sW + 1;
  y = make_f32({N, M, oH, oW});

  const float* xp = x.f32.data();
  const float* wp = w.f32.data();
  float* yp = y.f32.data();
  for (int64_t ni = 0; ni < N; ++ni) {
    for (int64_t mi = 0; mi < M; ++mi) {
      for (int64_t oi = 0; oi < oH; ++oi) {
        for (int64_t oj = 0; oj < oW; ++oj) {
          float acc = b ? b->f32[static_cast<std::size_t>(mi)] : 0.0f;
          for (int64_t ci = 0; ci < C; ++ci) {
            for (int64_t ki = 0; ki < kH; ++ki) {
              int64_t ih = oi * sH + ki * dH - pH;
              if (ih < 0 || ih >= H) continue;
              for (int64_t kj = 0; kj < kW; ++kj) {
                int64_t iw = oj * sW + kj * dW - pW;
                if (iw < 0 || iw >= W) continue;
                acc += xp[((ni * C + ci) * H + ih) * W + iw] *
                       wp[(((mi * C + ci) * kH) + ki) * kW + kj];
              }
            }
          }
          yp[((ni * M + mi) * oH + oi) * oW + oj] = acc;
          flops_hint_ += C * kH * kW;
        }
      }
    }
  }
  return true;
}

bool Engine::op_qlinear_conv(const Node& n, std::string& error) {
  // inputs: x, x_scale, x_zero_point, w, w_scale, w_zero_point,
  //         y_scale, y_zero_point [, B]
  if (n.inputs.size() < 8) {
    error = "qlinearconv expects >= 8 inputs";
    return false;
  }
  const Tensor* x = fetch(n.inputs[0]);
  const Tensor* xs = fetch(n.inputs[1]);
  const Tensor* w = fetch(n.inputs[3]);
  const Tensor* ws = fetch(n.inputs[4]);
  const Tensor* ys = fetch(n.inputs[6]);
  const Tensor* b = n.inputs.size() > 8 ? fetch(n.inputs[8]) : nullptr;
  if (!x || !xs || !w || !ws || !ys) {
    error = "qlinearconv missing operand";
    return false;
  }
  float x_scale = xs->f32.empty() ? 0.f : xs->f32[0];
  float w_scale = ws->f32.size() == 1 ? ws->f32[0] : -1.0f;  // -1 => per-channel
  float y_scale = ys->f32.empty() ? 0.f : ys->f32[0];

  auto pads = attr_ints(n, "pads");
  auto strides = attr_ints(n, "strides");
  if (pads.size() != 4) pads = {0, 0, 0, 0};
  if (strides.size() != 2) strides = {1, 1};

  const int32_t x_zp_i = zp_of(fetch(n.inputs[2]));
  const int32_t wzp_i = zp_of(fetch(n.inputs[5]));
  const Tensor* yzp = fetch(n.inputs[7]);
  const int32_t y_zp_i = zp_of(yzp);
  const bool y_u8 = yzp && yzp->u8;

  const int64_t N = x->n(), C = x->c(), H = x->h(), W = x->w();
  const int64_t M = w->dims[0], Cw = w->dims[1], kH = w->dims[2], kW = w->dims[3];

  const int64_t sH = strides[0], sW = strides[1];
  const int64_t oH = (H + pads[0] + pads[2] - kH) / sH + 1;
  const int64_t oW = (W + pads[1] + pads[3] - kW) / sW + 1;

  Tensor y;
  y.kind = Tensor::Kind::INT8;
  y.dims = {N, M, oH, oW};
  y.i8.assign(static_cast<std::size_t>(y.numel()), 0);
  for (int64_t ni = 0; ni < N; ++ni) {
    for (int64_t mi = 0; mi < M; ++mi) {
      float w_s = w_scale < 0 ? ws->f32[static_cast<std::size_t>(mi)] : w_scale;
      for (int64_t oi = 0; oi < oH; ++oi) {
        for (int64_t oj = 0; oj < oW; ++oj) {
          int32_t acc = 0;
          for (int64_t ci = 0; ci < C; ++ci) {
            for (int64_t ki = 0; ki < kH; ++ki) {
              int64_t ih = oi * sH + ki - pads[0];
              if (ih < 0 || ih >= H) continue;
              for (int64_t kj = 0; kj < kW; ++kj) {
                int64_t iw = oj * sW + kj - pads[1];
                if (iw < 0 || iw >= W) continue;
                int32_t xv = val_of(x, static_cast<std::size_t>(((ni * C + ci) * H + ih) * W + iw)) - x_zp_i;
                int32_t wv = val_of(w, static_cast<std::size_t>((((mi * Cw + ci) * kH) + ki) * kW + kj)) - wzp_i;
                acc += xv * wv;
              }
            }
          }
          if (b) acc += b->i32[static_cast<std::size_t>(mi)];  // int32 bias
          float real = static_cast<float>(acc) * x_scale * w_s;
          float q = real / y_scale + static_cast<float>(y_zp_i);
          int32_t qi = static_cast<int32_t>(std::lround(q));
          y.u8 = y_u8;
          y.i8[static_cast<std::size_t>(((ni * M + mi) * oH + oi) * oW + oj)] =
              static_cast<int8_t>(y_u8 ? static_cast<uint32_t>(std::clamp<int32_t>(qi, 0, 255))
                                       : std::clamp<int32_t>(qi, -128, 127));
          flops_hint_ += C * kH * kW;
        }
      }
    }
  }
  *store(n.outputs[0]) = std::move(y);
  return true;
}

bool Engine::op_matmul(const Tensor& a, const Tensor& b, Tensor& y) {
  // 2d x 2d only (what our graphs contain)
  if (a.dims.size() != 2 || b.dims.size() != 2) return false;
  Tensor out = make_f32({a.dims[0], b.dims[1]});
  for (int64_t i = 0; i < a.dims[0]; ++i) {
    for (int64_t k = 0; k < a.dims[1]; ++k) {
      float av = a.f32[static_cast<std::size_t>(i * a.dims[1] + k)];
      if (av == 0.0f) continue;
      for (int64_t j = 0; j < b.dims[1]; ++j) {
        out.f32[static_cast<std::size_t>(i * b.dims[1] + j)] +=
            av * b.f32[static_cast<std::size_t>(k * b.dims[1] + j)];
      }
    }
  }
  y = std::move(out);
  return true;
}

bool Engine::op_qlinear_matmul(const Node& n, std::string& error) {
  // inputs: a, a_scale, a_zp, b, b_scale, b_zp, y_scale, y_zp [, none]
  if (n.inputs.size() < 8) {
    error = "qlinearmatmul expects >= 8 inputs";
    return false;
  }
  const Tensor* a = fetch(n.inputs[0]);
  const Tensor* b = fetch(n.inputs[3]);
  const Tensor* as = fetch(n.inputs[1]);
  const Tensor* bs = fetch(n.inputs[4]);
  const Tensor* ys = fetch(n.inputs[6]);
  const Tensor* yzp = fetch(n.inputs[7]);
  if (!a || !b || !as || !bs || !ys) {
    error = "qlinearmatmul missing operand";
    return false;
  }
  const float a_s = as->f32.empty() ? 1.f : as->f32[0];
  const float b_s = bs->f32.empty() ? 1.f : bs->f32[0];
  const float y_s = ys->f32.empty() ? 1.f : ys->f32[0];
  const int32_t a_zp = zp_of(fetch(n.inputs[2]));
  const int32_t b_zp = zp_of(fetch(n.inputs[5]));
  const int32_t y_zp = zp_of(yzp);
  const bool y_u8 = yzp && yzp->u8;
  if (a->dims.size() != 2 || b->dims.size() != 2) {
    error = "qlinearmatmul 2d only";
    return false;
  }
  Tensor y;
  y.kind = Tensor::Kind::INT8;
  y.u8 = y_u8;
  y.dims = {a->dims[0], b->dims[1]};
  y.i8.assign(static_cast<std::size_t>(y.numel()), 0);
  for (int64_t i = 0; i < a->dims[0]; ++i) {
    for (int64_t j = 0; j < b->dims[1]; ++j) {
      int32_t acc = 0;
      for (int64_t k = 0; k < a->dims[1]; ++k) {
        acc += (val_of(a, static_cast<std::size_t>(i * a->dims[1] + k)) - a_zp) *
               (val_of(b, static_cast<std::size_t>(k * b->dims[1] + j)) - b_zp);
      }
      const float real = static_cast<float>(acc) * a_s * b_s;
      const int32_t q = static_cast<int32_t>(std::lround(real / y_s)) + y_zp;
      y.i8[static_cast<std::size_t>(i * b->dims[1] + j)] =
          static_cast<int8_t>(y_u8 ? static_cast<uint32_t>(std::clamp<int32_t>(q, 0, 255))
                                   : std::clamp<int32_t>(q, -128, 127));
    }
  }
  *store(n.outputs[0]) = std::move(y);
  return true;
}

bool Engine::op_maxpool(const Node& n, const Tensor* x, Tensor& y) {
  auto ks = attr_ints(n, "kernel_shape");
  auto strides = attr_ints(n, "strides");
  auto pads = attr_ints(n, "pads");
  if (ks.size() != 2) return false;
  if (strides.empty()) strides = ks;
  if (pads.size() != 4) pads = {0, 0, 0, 0};
  const int64_t N = x->n(), C = x->c(), H = x->h(), W = x->w();
  const int64_t kH = ks[0], kW = ks[1], sH = strides[0], sW = strides[1];
  const int64_t oH = (H + pads[0] + pads[2] - kH) / sH + 1;
  const int64_t oW = (W + pads[1] + pads[3] - kW) / sW + 1;
  // build the output fresh: copying the input first would double the peak
  // (a 22 kb int8 activation cannot afford that on device).
  const bool is_i8 = x->kind == Tensor::Kind::INT8;
  y.kind = x->kind;
  y.u8 = x->u8;
  y.dims = {N, C, oH, oW};
  if (is_i8) {
    y.i8.assign(static_cast<std::size_t>(y.numel()), 0);
  } else {
    y.f32.assign(static_cast<std::size_t>(y.numel()), 0.0f);
  }
  auto at = [&](int64_t idx) -> float {
    return is_i8 ? static_cast<float>(x->i8[static_cast<std::size_t>(idx)])
                 : x->f32[static_cast<std::size_t>(idx)];
  };
  auto put = [&](int64_t idx, float v) {
    if (is_i8) y.i8[static_cast<std::size_t>(idx)] = static_cast<int8_t>(v);
    else y.f32[static_cast<std::size_t>(idx)] = v;
  };
  for (int64_t ni = 0; ni < N; ++ni) {
    for (int64_t ci = 0; ci < C; ++ci) {
      for (int64_t oi = 0; oi < oH; ++oi) {
        for (int64_t oj = 0; oj < oW; ++oj) {
          float mx = -std::numeric_limits<float>::infinity();
          for (int64_t ki = 0; ki < kH; ++ki) {
            int64_t ih = oi * sH + ki - pads[0];
            if (ih < 0 || ih >= H) continue;
            for (int64_t kj = 0; kj < kW; ++kj) {
              int64_t iw = oj * sW + kj - pads[1];
              if (iw < 0 || iw >= W) continue;
              float v = at(((ni * C + ci) * H + ih) * W + iw);
              if (v > mx) mx = v;
            }
          }
          put(((ni * C + ci) * oH + oi) * oW + oj, mx);
        }
      }
    }
  }
  return true;
}

bool Engine::op_quantize_linear(const Node& n, std::string& error) {
  if (n.inputs.size() < 2) {
    error = "quantizelinear expects x, scale";
    return false;
  }
  const Tensor* x = fetch(n.inputs[0]);
  const Tensor* s = fetch(n.inputs[1]);
  const Tensor* zp = n.inputs.size() > 2 ? fetch(n.inputs[2]) : nullptr;
  if (!x || !s || x->kind != Tensor::Kind::F32) {
    error = "quantizelinear bad operand";
    return false;
  }
  float scale = s->f32.empty() ? 1.f : s->f32[0];
  const bool zu8 = zp && zp->u8;
  int32_t z = zp_of(zp);
  Tensor y;
  y.kind = Tensor::Kind::INT8;
  y.u8 = zu8;
  y.dims = x->dims;
  y.i8.resize(x->f32.size());
  for (std::size_t i = 0; i < x->f32.size(); ++i) {
    int32_t q = static_cast<int32_t>(std::lround(x->f32[i] / scale)) + z;
    y.i8[i] = static_cast<int8_t>(zu8 ? static_cast<uint32_t>(std::clamp<int32_t>(q, 0, 255))
                                     : std::clamp<int32_t>(q, -128, 127));
  }
  *store(n.outputs[0]) = std::move(y);
  return true;
}

bool Engine::op_dequantize_linear(const Node& n, std::string& error) {
  if (n.inputs.size() < 2) {
    error = "dequantizelinear expects x, scale";
    return false;
  }
  const Tensor* x = fetch(n.inputs[0]);
  const Tensor* s = fetch(n.inputs[1]);
  const Tensor* zp = n.inputs.size() > 2 ? fetch(n.inputs[2]) : nullptr;
  if (!x || !s) {
    error = "dequantizelinear bad operand";
    return false;
  }
  const bool per_channel = s->f32.size() > 1;   // scales along `axis` (default 1)
  const int64_t axis = attr_int(n, "axis", 1);
  if (per_channel && (axis < 0 || static_cast<std::size_t>(axis) >= x->dims.size())) {
    error = "dequantizelinear bad axis";
    return false;
  }
  // stride between consecutive channel slices along axis
  int64_t chan = per_channel ? x->dims[static_cast<std::size_t>(axis)] : 1;
  int64_t block = 1;
  for (std::size_t k = static_cast<std::size_t>(axis) + 1; k < x->dims.size(); ++k) {
    block *= x->dims[k];
  }
  auto scale_of = [&](std::size_t i) {
    if (!per_channel) return s->f32.empty() ? 1.f : s->f32[0];
    std::size_t c = static_cast<std::size_t>(i / block) % static_cast<std::size_t>(chan);
    return s->f32[c < s->f32.size() ? c : 0];
  };

  Tensor y = make_f32(x->dims);
  if (x->kind == Tensor::Kind::INT8) {
    for (std::size_t i = 0; i < x->i8.size(); ++i)
      y.f32[i] = static_cast<float>(val_of(x, i) - zp_of(zp)) * scale_of(i);
  } else if (x->kind == Tensor::Kind::INT32) {
    for (std::size_t i = 0; i < x->i32.size(); ++i)
      y.f32[i] = static_cast<float>(x->i32[i] - zp_of(zp)) * scale_of(i);
  } else {
    error = "dequantizelinear unsupported input dtype";
    return false;
  }
  *store(n.outputs[0]) = std::move(y);
  return true;
}


bool Engine::exec_node(const Node& n, std::string& error) {
  const std::string& op = n.op;

  if (getenv("GJ_TRACE")) {
    const Tensor* tin = fetch(n.inputs[0]);
    fprintf(stderr, "[trace] %s in=%s(%s) -> %s\n", op.c_str(), n.inputs[0].c_str(),
            tin ? tin->shape_str().c_str() : "?", n.outputs[0].c_str());
  }
  if (op == "Relu" || op == "Sigmoid" || op == "Clip" || op == "Sub" || op == "Div" ||
      op == "Add" || op == "Min" || op == "Max") {
    const Tensor* a = fetch(n.inputs[0]);
    if (!a) {
      error = op + ": missing input " + n.inputs[0];
      return false;
    }
    // in-place when the operand lives in the buffer map (relu on a 22 kb
    // int8 activation cannot afford a second copy on a 520 kb part).
    Tensor y;
    bool inplaced = false;
    auto ait = buffers_.find(n.inputs[0]);
    if ((op == "Relu" || op == "Clip") && ait != buffers_.end()) {
      y = std::move(ait->second);
      inplaced = true;
    } else {
      y = *a;
    }
    if (op == "Relu") {
      if (y.kind == Tensor::Kind::INT8) {
        // symmetric quantized activations: zero point is 0, relu is a clamp
        for (auto& v : y.i8) v = v > 0 ? v : static_cast<int8_t>(0);
      } else {
        for (auto& v : y.f32) v = v > 0.0f ? v : 0.0f;
      }
    } else if (op == "Sigmoid") {
      for (auto& v : y.f32) v = 1.0f / (1.0f + std::exp(-v));
    } else if (op == "Clip") {
      float lo = attr_f(n, "min", -3.4e38f);
      float hi = attr_f(n, "max", 3.4e38f);
      for (auto& v : y.f32) v = std::clamp(v, lo, hi);
    } else {
      const Tensor* b = fetch(n.inputs[1]);
      if (!b) {
        error = op + ": missing input " + n.inputs[1];
        return false;
      }
      if (a->dims.size() > 4) {
        error = op + ": rank > 4 unsupported";
        return false;
      }
      // right-align b into a's 4 dims; each dim must be 1 or equal
      std::array<int64_t, 4> ad{1, 1, 1, 1};
      for (std::size_t i = 0; i < a->dims.size(); ++i) ad[4 - a->dims.size() + i] = a->dims[i];
      std::array<int64_t, 4> bd{1, 1, 1, 1};
      std::array<int64_t, 4> bs{0, 0, 0, 0};   // 0 stride where b broadcasts
      for (std::size_t i = 0; i < b->dims.size(); ++i) {
        std::size_t slot = 4 - b->dims.size() + i;
        if (b->dims[i] != 1 && b->dims[i] != ad[slot]) {
          error = op + ": broadcast mismatch";
          return false;
        }
        bd[slot] = b->dims[i];
      }
      int64_t acc = 1;
      for (int k = 3; k >= 0; --k) {
        if (bd[k] != 1) { bs[k] = acc; acc *= bd[k]; }
      }
      std::size_t out_i = 0;
      for (int64_t i0 = 0; i0 < ad[0]; ++i0)
        for (int64_t i1 = 0; i1 < ad[1]; ++i1)
          for (int64_t i2 = 0; i2 < ad[2]; ++i2)
            for (int64_t i3 = 0; i3 < ad[3]; ++i3, ++out_i) {
              std::size_t bi = static_cast<std::size_t>(i0 * bs[0] + i1 * bs[1] + i2 * bs[2] + i3 * bs[3]);
              float bv = b->f32[bi];
              float av = y.f32[out_i];
              if (op == "Sub") y.f32[out_i] = av - bv;
              else if (op == "Div") y.f32[out_i] = av / bv;
              else if (op == "Add") y.f32[out_i] = av + bv;
              else if (op == "Min") y.f32[out_i] = av < bv ? av : bv;
              else y.f32[out_i] = av > bv ? av : bv;  // Max
            }
    }
    *store(n.outputs[0]) = std::move(y);
    (void)inplaced;  // the liveness pass erases the (now empty) input name
    return true;
  }

  if (op == "Conv") {
    const Tensor* x = fetch(n.inputs[0]);
    const Tensor* w = fetch(n.inputs[1]);
    const Tensor* b = n.inputs.size() > 2 ? fetch(n.inputs[2]) : nullptr;
    if (!x || !w || x->kind != Tensor::Kind::F32 || w->kind != Tensor::Kind::F32) {
      error = "conv: missing or non-float operand";
      return false;
    }
    Tensor y;
    if (!op_conv(n, *x, *w, b, y, error)) return false;
    *store(n.outputs[0]) = std::move(y);
    return true;
  }
  if (op == "QLinearConv") return op_qlinear_conv(n, error);
  if (op == "QLinearMatMul") return op_qlinear_matmul(n, error);

  if (op == "MatMul") {
    const Tensor* a = fetch(n.inputs[0]);
    const Tensor* b = fetch(n.inputs[1]);
    if (!a || !b || a->kind != Tensor::Kind::F32 || b->kind != Tensor::Kind::F32) {
      error = "matmul: bad operand";
      return false;
    }
    Tensor y;
    if (!op_matmul(*a, *b, y)) {
      error = "matmul: rank > 2 unsupported";
      return false;
    }
    *store(n.outputs[0]) = std::move(y);
    return true;
  }

  if (op == "MaxPool") {
    const Tensor* x = fetch(n.inputs[0]);
    if (!x) {
      error = "maxpool: missing input";
      return false;
    }
    Tensor y;
    if (!op_maxpool(n, x, y)) {
      error = "maxpool: bad attributes";
      return false;
    }
    *store(n.outputs[0]) = std::move(y);
    return true;
  }

  if (op == "GlobalAveragePool") {
    const Tensor* x = fetch(n.inputs[0]);
    if (!x || x->dims.size() != 4) {
      error = "gap: missing or non-4d input";
      return false;
    }
    Tensor y = make_f32({x->n(), x->c(), 1, 1});
    const int64_t area = x->h() * x->w();
    for (int64_t ni = 0; ni < x->n(); ++ni) {
      for (int64_t ci = 0; ci < x->c(); ++ci) {
        float acc = 0.0f;
        const float* src = x->f32.data() + ((ni * x->c() + ci) * area);
        for (int64_t i = 0; i < area; ++i) acc += src[i];
        y.f32[static_cast<std::size_t>(ni * x->c() + ci)] = acc / static_cast<float>(area);
      }
    }
    *store(n.outputs[0]) = std::move(y);
    return true;
  }

  if (op == "QuantizeLinear") return op_quantize_linear(n, error);
  if (op == "DequantizeLinear") return op_dequantize_linear(n, error);

  if (op == "Reshape") {
    const Tensor* x = fetch(n.inputs[0]);
    const Tensor* shape = n.inputs.size() > 1 ? fetch(n.inputs[1]) : nullptr;
    if (!x || !shape || shape->kind != Tensor::Kind::INT64) {
      error = "reshape: missing input or non-i64 shape";
      return false;
    }
    Tensor y = *x;
    std::vector<int64_t> dims;
    int64_t infer_idx = -1, known = 1;
    for (std::size_t i = 0; i < shape->i64.size(); ++i) {
      int64_t d = shape->i64[i];
      if (d == -1) { infer_idx = static_cast<int64_t>(i); dims.push_back(1); }
      else { dims.push_back(d); if (d > 0) known *= d; }
    }
    if (infer_idx >= 0) {
      dims[static_cast<std::size_t>(infer_idx)] = x->numel() / std::max<int64_t>(known, 1);
    }
    y.dims = std::move(dims);
    *store(n.outputs[0]) = std::move(y);
    return true;
  }

  if (op == "Identity") {
    const Tensor* x = fetch(n.inputs[0]);
    if (!x) {
      error = "identity: missing input";
      return false;
    }
    *store(n.outputs[0]) = *x;
    return true;
  }

  error = "exec: unhandled op " + op;
  return false;
}

}  // namespace gj::onnx
