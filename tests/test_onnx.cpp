#include "doctest.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "../src/onnx/engine.hpp"
#include "golden/model_fixtures.hpp"

namespace {
std::vector<std::byte> slurp(const std::string& path) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) return {};
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  std::vector<std::byte> out(static_cast<std::size_t>(sz));
  size_t got = fread(out.data(), 1, out.size(), f);
  fclose(f);
  out.resize(got);
  return out;
}

float run_one(gj::onnx::Engine& eng, const float* spec, std::string& err) {
  gj::onnx::Tensor in;
  in.kind = gj::onnx::Tensor::Kind::F32;
  in.dims = {1, 1, kSpecBands, kSpecFrames};
  in.f32.assign(spec, spec + kSpecBands * kSpecFrames);
  std::vector<gj::onnx::Tensor> outs;
  if (!eng.run({in}, outs, err) || outs.empty()) return -1.0f;
  return outs.back().f32.empty() ? -1.0f : outs.back().f32[0];
}

const float* golden_in(int i) {
  switch (i) {
    case 0: return kIn0; case 1: return kIn1; case 2: return kIn2; case 3: return kIn3;
    case 4: return kIn4; case 5: return kIn5; case 6: return kIn6; default: return kIn7;
  }
}
float golden_out_f(int i) {
  switch (i) {
    case 0: return kOutF0; case 1: return kOutF1; case 2: return kOutF2; case 3: return kOutF3;
    case 4: return kOutF4; case 5: return kOutF5; case 6: return kOutF6; default: return kOutF7;
  }
}
float golden_out_q(int i) {
  switch (i) {
    case 0: return kOutQ0; case 1: return kOutQ1; case 2: return kOutQ2; case 3: return kOutQ3;
    case 4: return kOutQ4; case 5: return kOutQ5; case 6: return kOutQ6; default: return kOutQ7;
  }
}
}  // namespace

TEST_CASE("onnx engine: parses the exported graph") {
  auto bytes = slurp(GLASSJAW_MODELS_DIR "/detector.onnx");
  REQUIRE_FALSE(bytes.empty());
  gj::onnx::Engine eng;
  std::string err;
  REQUIRE(eng.load(bytes, err));
  MESSAGE("ops parsed ok");
}

TEST_CASE("onnx engine: float model matches onnxruntime") {
  auto bytes = slurp(GLASSJAW_MODELS_DIR "/detector.onnx");
  REQUIRE_FALSE(bytes.empty());
  gj::onnx::Engine eng;
  std::string err;
  REQUIRE(eng.load(bytes, err));
  for (int i = 0; i < kGoldenCount; ++i) {
    float got = run_one(eng, golden_in(i), err);
    float want = golden_out_f(i);
    CHECK_MESSAGE(std::fabs(got - want) < 3e-3,
                  "case " << i << " got " << got << " want " << want << " err " << err);
  }
}

TEST_CASE("onnx engine: int8 (qlinear) model matches onnxruntime") {
  auto bytes = slurp(GLASSJAW_MODELS_DIR "/detector_int8.onnx");
  REQUIRE_FALSE(bytes.empty());
  gj::onnx::Engine eng;
  std::string err;
  REQUIRE(eng.load(bytes, err));
  for (int i = 0; i < kGoldenCount; ++i) {
    float got = run_one(eng, golden_in(i), err);
    float want = golden_out_q(i);
    CHECK_MESSAGE(std::fabs(got - want) < 0.02,
                  "case " << i << " got " << got << " want " << want << " err " << err);
  }
}
