#include "doctest.h"

#include <cmath>
#include <vector>

#include "../src/dsp.hpp"
#include "golden/dsp_fixtures.hpp"

using namespace gj;

namespace {
bool close_enough(float a, float b, float rel = 2e-3f, float abs = 5e-3f) {
  float diff = std::fabs(a - b);
  float scale = std::max(std::fabs(a), std::fabs(b));
  return diff <= abs || diff <= rel * scale;
}
}  // namespace

TEST_CASE("dsp: spectrogram matches python golden vectors") {
  for (int c = 0; c < kCaseCount; ++c) {
    std::vector<int16_t> pcm(kWindow);
    const float* src = nullptr;
    switch (c) {
      case 0: src = kCase0_PcmAsFloat; break;
      case 1: src = kCase1_PcmAsFloat; break;
      case 2: src = kCase2_PcmAsFloat; break;
      case 3: src = kCase3_PcmAsFloat; break;
    }
    for (int i = 0; i < kWindow; ++i) pcm[i] = static_cast<int16_t>(src[i]);
    std::vector<float> spec(kSpecLen);
    auto res = gj::log_mel_spectrogram(pcm, spec);

    const float* want = nullptr;
    float want_peak = 0;
    switch (c) {
      case 0: want = kCase0_Spec; want_peak = kCase0_PeakDb; break;
      case 1: want = kCase1_Spec; want_peak = kCase1_PeakDb; break;
      case 2: want = kCase2_Spec; want_peak = kCase2_PeakDb; break;
      case 3: want = kCase3_Spec; want_peak = kCase3_PeakDb; break;
    }
    int bad = 0;
    float worst = 0;
    for (int i = 0; i < kSpecLen; ++i) {
      float d = std::fabs(spec[i] - want[i]);
      if (d > worst) worst = d;
      if (!close_enough(spec[i], want[i])) ++bad;
    }
    CHECK(bad == 0);
    CHECK(worst < 0.05f);
    CHECK(close_enough(res.peak_dbfs, want_peak, 1e-4f, 1e-3f));
  }
}

TEST_CASE("dsp: silence produces the log floor everywhere") {
  std::vector<int16_t> pcm(kWindowSamples, 0);
  std::vector<float> spec(kMelBands * kFramesPerWindow);
  auto res = gj::log_mel_spectrogram(pcm, spec);
  CHECK(res.peak_dbfs < -200.0f);
  for (float v : spec) CHECK(v == doctest::Approx(std::log10(1e-10f)));
}

TEST_CASE("dsp: pure tone lands in the expected mel band") {
  std::vector<int16_t> pcm(kWindowSamples);
  for (int i = 0; i < kWindowSamples; ++i) {
    pcm[i] = static_cast<int16_t>(12000.0f * std::sin(2.0f * 3.14159265f * 1000.0f * i / 16000.0f));
  }
  std::vector<float> spec(kMelBands * kFramesPerWindow);
  gj::log_mel_spectrogram(pcm, spec);
  // find the hottest band at frame 30
  int best = 0;
  float best_v = -1e30f;
  for (int b = 0; b < kMelBands; ++b) {
    float v = spec[b * kFramesPerWindow + 30];
    if (v > best_v) { best_v = v; best = b; }
  }
  // 1 kHz sits around band 9-11 of 26 spanning 50..8000 hz (htk mel)
  CHECK(best >= 8);
  CHECK(best <= 12);
}
