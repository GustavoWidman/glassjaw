#include "pipeline.hpp"

#include "../dsp.hpp"

#include <chrono>
#include <cstdio>

#if defined(ESP_PLATFORM)
#include "esp_timer.h"
#endif

namespace gj {

int64_t Pipeline::micros() {
#if defined(ESP_PLATFORM)
  return static_cast<int64_t>(esp_timer_get_time());
#else
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
#endif
}

Pipeline::Pipeline(MicSource& mic, AlertSink& alerts, std::span<const std::byte> model,
                   float threshold, const PipelineHooks& hooks)
    : mic_(mic), alerts_(alerts), hooks_(hooks), detector_(threshold) {
  // boot-time allocations: contiguous blocks this big only exist before the
  // tasks and their stacks carve up the heap.
  ring_ = new SpscRing<int16_t, 32768>();
  spec_pool_ = new Spectrogram[kFeatureQueueDepth + 1];
  std::string err;
  engine_ok_ = engine_.load(model, err);
  if (!engine_ok_) {
    dprintf(STDERR_FILENO, "glassjaw: model load failed: %s\n", err.c_str());
  }
}

bool Pipeline::start(BaseType_t core) {
#if !defined(ESP_PLATFORM)
  (void)core;  // pinning is an esp-idf concept; upstream kernel ignores it
#endif
  if (!engine_ok_) return false;
  // created here, not in the constructor: the posix port forbids kernel
  // calls before the scheduler starts.
  window_sem_ = xSemaphoreCreateBinary();
  spec_queue_ = xQueueCreate(kFeatureQueueDepth, sizeof(SpecMsg));
  hop_queue_ = xQueueCreate(16, sizeof(uint32_t));
  stats_mutex_ = xSemaphoreCreateMutex();
  BaseType_t ok = pdPASS;
#if defined(ESP_PLATFORM)
  // esp-idf: stack depth in bytes, optional core pinning
  #define GJ_TASK_CREATE(fn, name, stack, arg, prio) \
    xTaskCreatePinnedToCore(fn, name, stack, arg, prio, nullptr, core)
#else
  // upstream kernel + posix port: stack depth semantics are the port's
  #define GJ_TASK_CREATE(fn, name, stack, arg, prio) \
    xTaskCreate(fn, name, stack, arg, prio, nullptr)
#endif
  ok &= GJ_TASK_CREATE(capture_entry, "capture", kCaptureStack, this, kCapturePrio);
  ok &= GJ_TASK_CREATE(features_entry, "features", kFeaturesStack, this, kFeaturesPrio);
  ok &= GJ_TASK_CREATE(detect_entry, "detect", kDetectStack, this, kDetectPrio);
  ok &= GJ_TASK_CREATE(monitor_entry, "monitor", kMonitorStack, this, kMonitorPrio);
  return ok == pdPASS;
}

void Pipeline::request_stop() { stop_.store(true, std::memory_order_relaxed); }

void Pipeline::record(LatencyStats<256>& slot, int64_t us) {
  if (xSemaphoreTake(stats_mutex_, portMAX_DELAY) == pdTRUE) {
    slot.push(us);
    xSemaphoreGive(stats_mutex_);
  }
}

// ---- capture: mic -> ring ------------------------------------------------

void Pipeline::capture_entry(void* arg) { static_cast<Pipeline*>(arg)->capture_loop(); }

void Pipeline::capture_loop() {
  std::array<int16_t, kChunk> chunk{};
  while (!stop_.load(std::memory_order_relaxed)) {
    int64_t t0 = micros();
    std::size_t n = mic_.read(chunk);
    if (n == 0) {  // end of stream (simulator)
      stop_.store(true, std::memory_order_relaxed);
      xSemaphoreGive(window_sem_);   // release the feature task
      SpecMsg msg{nullptr, -240.0f, 0, 0};
      xQueueSend(spec_queue_, &msg, 0);  // release the detect task
      break;
    }
    ring_->write(std::span<const int16_t>(chunk.data(), n));
    alerts_.tick(detector_.alarm());
    hop_credit_ += static_cast<int>(n);
    if (hop_credit_ >= kWindowHop) {
      hop_credit_ -= kWindowHop;
      window_t0_.store(micros(), std::memory_order_relaxed);  // e2e clock start
      const uint32_t hop = static_cast<uint32_t>(total_hops_++) + 1;
      uint32_t stale_hop = 0;
      if (xQueueSend(hop_queue_, &hop, 0) != pdTRUE) {
        // 16 hops of lag: drop the oldest, the newest evidence wins
        if (xQueueReceive(hop_queue_, &stale_hop, 0) == pdTRUE) ++dropped_hops_;
        xQueueSend(hop_queue_, &hop, 0);
      }
      xSemaphoreGive(window_sem_);
    }
    int64_t us = micros() - t0;
    record(capture_us_, us);
    if (hooks_.on_capture) hooks_.on_capture(hooks_.user, us);
    // fast mode never blocks on the source; without a yield here the
    // priority-5 task starves everything below it.
    if (!mic_.realtime()) vTaskDelay(1);
  }
  vTaskDelete(nullptr);  // freertos tasks must never return
}

// ---- features: ring -> spectrogram ----------------------------------------

void Pipeline::features_entry(void* arg) { static_cast<Pipeline*>(arg)->features_loop(); }

void Pipeline::features_loop() {
  // task-static: too large for the task stack, single caller.
  static std::array<int16_t, kWindowSamples> window{};
  for (;;) {
    if (xSemaphoreTake(window_sem_, portMAX_DELAY) != pdTRUE) continue;
    // keep processing captured windows after stop; the watcher ends the
    // scheduler when the queues drain.
    if (stop_ && ring_->available() < kWindowSamples) break;
    int64_t t0 = micros();
    ring_->peek_newest(window);

    // spectrogram pool: queue slots + one in flight, so a slot is never
    // reused while still queued.
    static uint32_t next = 0;
    auto* spec = &spec_pool_[next++ % (kFeatureQueueDepth + 1)];
    auto res = log_mel_spectrogram(window, std::span<float>(spec->data(), spec->size()));

    int64_t wt0 = window_t0_.load(std::memory_order_relaxed);
    uint32_t hop = 0;
    if (xQueueReceive(hop_queue_, &hop, 0) != pdTRUE) hop = 0;  // desync: warmup
    SpecMsg msg{spec, res.peak_dbfs, wt0 ? wt0 : t0, hop};
    if (xQueueSend(spec_queue_, &msg, 0) != pdTRUE) {
      // queue full: drop the oldest pending window.
      SpecMsg stale{};
      if (xQueueReceive(spec_queue_, &stale, 0) == pdTRUE) { /* dropped */ }
      xQueueSend(spec_queue_, &msg, 0);
    }
    int64_t us = micros() - t0;
    record(feat_us_, us);
    if (hooks_.on_features) hooks_.on_features(hooks_.user, us);
  }
  vTaskDelete(nullptr);
}

// ---- detect: spectrogram -> onnx -> debounce -> alerts --------------------

void Pipeline::detect_entry(void* arg) { static_cast<Pipeline*>(arg)->detect_loop(); }

void Pipeline::detect_loop() {
  SpecMsg msg{};
  std::vector<onnx::Tensor> outs;
  bool alarm_prev = false;
  while (true) {
    if (xQueueReceive(spec_queue_, &msg, portMAX_DELAY) != pdTRUE) continue;
    if (msg.data == nullptr) break;  // shutdown marker
    int64_t t0 = micros();

    Detection det{};
    det.peak_dbfs = msg.peak_dbfs;
    det.gated = msg.peak_dbfs < kEnergyGateDb;
    det.hop = msg.hop;

    if (!det.gated) {
      onnx::Tensor in;
      in.kind = onnx::Tensor::Kind::F32;
      in.dims = {1, 1, kModelInBands, kModelInFrames};
      in.f32.assign(msg.data->begin(), msg.data->end());
      outs.clear();
      std::string err;
      if (engine_.run({in}, outs, err) && !outs.empty() && !outs.back().f32.empty()) {
        det.score = outs.back().f32[0];
      } else {
        dprintf(STDERR_FILENO, "glassjaw: inference failed: %s\n", err.c_str());
      }
    }
    det.alarm = detector_.update(det);

    if (det.alarm != alarm_prev) {
      alerts_.set_alarm(det.alarm);
      alarm_prev = det.alarm;
    }
    alerts_.heartbeat(StageTiming{0, 0, micros() - t0, micros() - t0}, det.score, det.alarm);

    int64_t us = micros() - t0;
    record(infer_us_, us);
    if (msg.ready_us) record(total_us_, micros() - msg.ready_us);
    if (hooks_.on_infer) hooks_.on_infer(hooks_.user, us);
    if (hooks_.on_detection) hooks_.on_detection(hooks_.user, det);
  }
  vTaskDelete(nullptr);
}

// ---- monitor: periodic latency report --------------------------------------

void Pipeline::monitor_entry(void* arg) { static_cast<Pipeline*>(arg)->monitor_loop(); }

void Pipeline::monitor_loop() {
  while (!stop_.load(std::memory_order_relaxed)) {
    vTaskDelay(pdMS_TO_TICKS(5000));
    if (stop_.load(std::memory_order_relaxed)) break;
    // snapshot percentiles under the mutex (each call sorts a small copy);
    // the uart print stays outside the critical section.
    int64_t v[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    if (xSemaphoreTake(stats_mutex_, portMAX_DELAY) == pdTRUE) {
      v[0] = capture_us_.percentile(0.5f); v[1] = capture_us_.percentile(0.99f);
      v[2] = feat_us_.percentile(0.5f);    v[3] = feat_us_.percentile(0.99f);
      v[4] = infer_us_.percentile(0.5f);   v[5] = infer_us_.percentile(0.99f);
      v[6] = total_us_.percentile(0.5f);   v[7] = total_us_.percentile(0.99f);
      xSemaphoreGive(stats_mutex_);
    }
#if defined(ESP_PLATFORM)
    printf("[monitor] capture p50=%lld p99=%lld | features p50=%lld p99=%lld | "
           "infer p50=%lld p99=%lld | e2e p50=%lld p99=%lld (us)\n",
           (long long)v[0], (long long)v[1], (long long)v[2], (long long)v[3],
           (long long)v[4], (long long)v[5], (long long)v[6], (long long)v[7]);
    fflush(stdout);
#else
    dprintf(STDERR_FILENO,
            "[monitor] capture p50=%lld p99=%lld | features p50=%lld p99=%lld | "
            "infer p50=%lld p99=%lld | e2e p50=%lld p99=%lld (us)\n",
            (long long)v[0], (long long)v[1], (long long)v[2], (long long)v[3],
            (long long)v[4], (long long)v[5], (long long)v[6], (long long)v[7]);
#endif
  }
  vTaskDelete(nullptr);
}

}  // namespace gj
