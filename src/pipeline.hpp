// shared constants and pipeline types. tools/features.py mirrors these
// values; the golden tests pin the two implementations together.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace gj {

// ---- audio front end -------------------------------------------------------
inline constexpr int kSampleRate = 16'000;        // Hz
inline constexpr int kFftSize = 512;              // 32 ms analysis frame
inline constexpr int kHop = 256;                  // 16 ms hop
inline constexpr int kMelBands = 26;
inline constexpr float kMelMinHz = 50.0f;
inline constexpr float kMelMaxHz = 8'000.0f;
inline constexpr float kLogFloor = 1e-10f;        // eps inside log10
inline constexpr int kFramesPerWindow = 61;       // 1 s of audio

// ---- inference window ------------------------------------------------------
inline constexpr int kWindowSamples = 16'000;     // 1.0 s
inline constexpr int kWindowHop = 8'000;          // 0.5 s, 50% overlap

// ---- model input/output ----------------------------------------------------
inline constexpr int kModelInBands = kMelBands;   // spectrogram H
inline constexpr int kModelInFrames = kFramesPerWindow;  // spectrogram W

// windows quieter than this are skipped (see report, model section).
inline constexpr float kEnergyGateDb = -40.0f;

// ---- detector --------------------------------------------------------------
// alarm fires when the smoothed score crosses the threshold and stays above
// it for kAlarmHold consecutive inferences; clears after kClearHold below.
inline constexpr int kAlarmHold = 2;
inline constexpr int kClearHold = 3;

// ---- rtos plumbing
inline constexpr int kFeatureQueueDepth = 3;      // spectrogram windows in flight
inline constexpr int kDetectionQueueDepth = 4;    // scores in flight

using Spectrogram = std::array<float, kMelBands * kFramesPerWindow>;

struct Detection {
  float score;         // sigmoid output of the model
  float peak_dbfs;     // window peak level at inference time
  bool gated;          // true when the energy gate skipped the model
  bool alarm;          // latched alarm state after debounce
  uint32_t hop;        // window index since boot (one hop = 500 ms of audio)
};

// per-stage latency sample in microseconds
struct StageTiming {
  int64_t capture_us;
  int64_t features_us;
  int64_t infer_us;
  int64_t total_us;
};

}  // namespace gj
