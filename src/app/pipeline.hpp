// freertos pipeline, shared by the esp32 firmware and the host simulator.
//
//   capture  prio 5: mic -> spsc ring, one semaphore give per 500 ms hop
//   features prio 4: newest 1 s window -> log-mel spectrogram -> queue
//   detect   prio 2: energy gate -> onnx engine -> debounce -> alerts
//   monitor  prio 1: latency percentiles every 5 s
#pragma once

#include <atomic>
#include <cstdint>
#include <string>

#include "freertos/FreeRTOS.h"

#include "../detector.hpp"
#include "../onnx/engine.hpp"
#include "../pipeline.hpp"
#include "../ring_buffer.hpp"
#include "../stats.hpp"

namespace gj {

// implemented by firmware (i2s) and simulator (wav files / synthetic sources)
class MicSource {
 public:
  virtual ~MicSource() = default;
  // block until at least one sample chunk is ready; fill dst; return count
  // (0 = end of stream, lets the simulator drain and exit cleanly).
  virtual std::size_t read(std::span<int16_t> dst) = 0;
  // true when the source should pace itself to wall clock (realtime mode)
  virtual bool realtime() const = 0;
};

class AlertSink {
 public:
  virtual ~AlertSink() = default;
  virtual void set_alarm(bool on) = 0;
  virtual void heartbeat(const StageTiming& t, float score, bool alarm) = 0;
};

struct PipelineHooks {
  // called by capture after each hop completes (t = time spent in read+push)
  void (*on_capture)(void* user, int64_t us) = nullptr;
  void (*on_features)(void* user, int64_t us) = nullptr;
  void (*on_infer)(void* user, int64_t us) = nullptr;
  void (*on_detection)(void* user, const Detection&) = nullptr;
  void* user = nullptr;
};

class Pipeline {
 public:
  Pipeline(MicSource& mic, AlertSink& alerts, std::span<const std::byte> model,
           float threshold, const PipelineHooks& hooks = {});

  // spins up all tasks; returns false if the model fails to load. on esp-idf
  // builds core >= 0 pins every task to that core (xTaskCreatePinnedToCore);
  // the host simulator passes the default -1.
  bool start(BaseType_t core = -1);
  void request_stop();
  bool stopped() const { return stop_; }

  const LatencyStats<256>& capture_stats() const { return capture_us_; }
  const LatencyStats<256>& features_stats() const { return feat_us_; }
  const LatencyStats<256>& infer_stats() const { return infer_us_; }
  const LatencyStats<256>& total_stats() const { return total_us_; }

  // monotonic microseconds (esp_timer on device, clock_gettime in sim)
  static int64_t micros();

  static constexpr UBaseType_t kCapturePrio = 5;
  static constexpr UBaseType_t kFeaturesPrio = 4;
  static constexpr UBaseType_t kDetectPrio = 2;
  static constexpr UBaseType_t kMonitorPrio = 1;

  static constexpr int kChunk = 1024;  // samples per mic read (~64 ms @ 16k)

 private:
  void record(LatencyStats<256>& slot, int64_t us);

  static void capture_entry(void* arg);
  static void features_entry(void* arg);
  static void detect_entry(void* arg);
  static void monitor_entry(void* arg);
  void capture_loop();
  void features_loop();
  void detect_loop();
  void monitor_loop();

  MicSource& mic_;
  // allocated once at boot (not static: bss overflowed), see pipeline.cpp.
  SpscRing<int16_t, 32768>* ring_{nullptr};
  Spectrogram* spec_pool_{nullptr};
  AlertSink& alerts_;
  PipelineHooks hooks_;
  onnx::Engine engine_;
  Detector detector_;
  bool engine_ok_ = false;

  struct SpecMsg {            // spectrogram + metadata measured with it
    Spectrogram* data;
    float peak_dbfs;
    int64_t ready_us;         // when the window completed capture (e2e clock)
    uint32_t hop;             // window index (0 = first full second of audio)
  };

  // esp-idf counts stack depth in bytes. the posix port ignores it.
  static constexpr uint32_t kCaptureStack = 8192;
  static constexpr uint32_t kFeaturesStack = 12288;
  static constexpr uint32_t kDetectStack = 12288;
  static constexpr uint32_t kMonitorStack = 4096;

  SemaphoreHandle_t window_sem_;   // given once per kWindowHop samples
  QueueHandle_t spec_queue_;      // SpecMsg windows in flight
  SemaphoreHandle_t stats_mutex_;
  std::atomic<bool> stop_{false};

  int hop_credit_ = 0;            // samples accumulated toward next window
  // capture pushes one hop index per semaphore give; features pops it with
  // the take. a shared counter would race when features lags behind capture
  // (the read would pick up a later hop than the window it just got).
  QueueHandle_t hop_queue_;
  uint32_t total_hops_ = 0;       // semaphore gives since boot
  uint32_t dropped_hops_ = 0;     // hop-queue overflows (extreme lag)

  // 64-bit value, 32-bit cpu: needs to be atomic or it tears.
  std::atomic<int64_t> window_t0_{0};

  LatencyStats<256> capture_us_, feat_us_, infer_us_, total_us_;
};

}  // namespace gj
