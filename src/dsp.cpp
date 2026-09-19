#include "dsp.hpp"

#include <algorithm>
#include <numbers>

namespace gj {

namespace {
constexpr int kBins = kFftSize / 2 + 1;

float hz_to_mel(float hz) { return 2595.0f * std::log10(1.0f + hz / 700.0f); }
[[maybe_unused]] float mel_to_hz(float mel) { return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f); }

// built once per process. the features task is the only caller;
// function-local statics make the init thread-safe.
struct Tables {
  std::array<float, kFftSize> win;
  std::array<std::array<float, kBins>, kMelBands> fb;
  Tables() {
    for (int n = 0; n < kFftSize; ++n) {
      win[n] = 0.54f - 0.46f * std::cos(2.0f * std::numbers::pi_v<float> * n / (kFftSize - 1));
    }
    const float mel_min = hz_to_mel(kMelMinHz);
    const float mel_max = hz_to_mel(kMelMaxHz);
    std::array<float, kMelBands + 2> pts{};
    for (int j = 0; j < kMelBands + 2; ++j) {
      pts[j] = mel_min + (mel_max - mel_min) * j / (kMelBands + 1);
    }
    for (int bin = 0; bin < kBins; ++bin) {
      const float m = hz_to_mel(bin * static_cast<float>(kSampleRate) / kFftSize);
      for (int b = 0; b < kMelBands; ++b) {
        const float up = (m - pts[b]) / (pts[b + 1] - pts[b]);
        const float down = (pts[b + 2] - m) / (pts[b + 2] - pts[b + 1]);
        fb[b][bin] = std::max(std::min(up, down), 0.0f);
      }
    }
  }
};

const Tables& tables() {
  static const Tables t;
  return t;
}
}  // namespace

std::array<float, kFftSize> make_hamming() { return tables().win; }

std::array<std::array<float, kFftSize / 2 + 1>, kMelBands> make_mel_bank() { return tables().fb; }

void fft(float* re, float* im, std::size_t n) {
  for (std::size_t i = 1, j = 0; i < n; ++i) {
    std::size_t bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      std::swap(re[i], re[j]);
      std::swap(im[i], im[j]);
    }
  }
  for (std::size_t len = 2; len <= n; len <<= 1) {
    const float ang = -2.0f * std::numbers::pi_v<float> / static_cast<float>(len);
    const float wr = std::cos(ang), wi = std::sin(ang);
    for (std::size_t i = 0; i < n; i += len) {
      float cr = 1.0f, ci = 0.0f;
      for (std::size_t k = 0; k < len / 2; ++k) {
        const float ur = re[i + k], ui = im[i + k];
        const float vr = re[i + k + len / 2] * cr - im[i + k + len / 2] * ci;
        const float vi = re[i + k + len / 2] * ci + im[i + k + len / 2] * cr;
        re[i + k] = ur + vr;
        im[i + k] = ui + vi;
        re[i + k + len / 2] = ur - vr;
        im[i + k + len / 2] = ui - vi;
        const float ncr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = ncr;
      }
    }
  }
}

float peak_dbfs(std::span<const int16_t> window) {
  int peak = 0;
  for (int16_t s : window) {
    const int v = s < 0 ? -static_cast<int>(s) : static_cast<int>(s);
    if (v > peak) peak = v;
  }
  if (peak == 0) return -240.0f;
  return 20.0f * std::log10(static_cast<float>(peak) / 32768.0f);
}

SpectrogramResult log_mel_spectrogram(std::span<const int16_t> window,
                                      std::span<float> spec_out) {
  const auto& tab = tables();
  // single caller, ~5 kb: static scratch.
  static float re[kFftSize];
  static float im[kFftSize];
  static float power[kBins];

  for (int f = 0; f < kFramesPerWindow; ++f) {
    const int off = f * kHop;
    for (int n = 0; n < kFftSize; ++n) {
      const float x = static_cast<float>(window[off + n]) / 32768.0f;
      re[n] = x * tab.win[n];
      im[n] = 0.0f;
    }
    fft(re, im, kFftSize);
    for (int b = 0; b < kBins; ++b) {
      power[b] = (re[b] * re[b] + im[b] * im[b]) / kFftSize;
    }
    for (int band = 0; band < kMelBands; ++band) {
      float acc = 0.0f;
      for (int b = 0; b < kBins; ++b) acc += tab.fb[band][b] * power[b];
      spec_out[band * kFramesPerWindow + f] = std::log10(acc + kLogFloor);
    }
  }
  return {peak_dbfs(window)};
}

}  // namespace gj
