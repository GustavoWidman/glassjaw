// glassjaw host simulator — runs the exact freertos pipeline the esp32 runs,
// fed by wav files instead of an i2s microphone.
//
// modes:
//   realtime  paces audio to the wall clock (validates scheduling behavior)
//   fast      drains as fast as the cpu allows (accuracy/latency harness)
//
// output: per-window csv on stdout (score/gate/alarm + per-stage timings) so
// tools/simulate.py can sweep a whole dataset and compute metrics.
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "app/pipeline.hpp"

namespace {

// ---- tiny riff/wave reader (16-bit pcm mono; resample offline) -------------
struct Wav {
  std::vector<int16_t> pcm;
  bool ok = false;
};

Wav read_wav(const std::string& path) {
  Wav w;
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) return w;
  char hdr[12];
  if (fread(hdr, 1, 12, f) != 12) { fclose(f); return w; }
  if (memcmp(hdr, "RIFF", 4) || memcmp(hdr + 8, "WAVE", 4)) { fclose(f); return w; }
  bool data_found = false, fmt_found = false;
  int channels = 1, bits = 16;
  while (!data_found) {
    char id[4];
    unsigned sz;
    if (fread(id, 1, 4, f) != 4 || fread(&sz, 4, 1, f) != 1) break;
    if (!memcmp(id, "fmt ", 4)) {
      short fmt, ch;
      int rate;
      unsigned br; short align;
      fread(&fmt, 2, 1, f); fread(&ch, 2, 1, f); fread(&rate, 4, 1, f);
      fread(&br, 4, 1, f); fread(&align, 2, 1, f); fread(&bits, 2, 1, f);
      (void)fmt; (void)rate; (void)br; (void)align;
      channels = ch;
      fmt_found = true;
      if (sz > 16) fseek(f, sz - 16, SEEK_CUR);
    } else if (!memcmp(id, "data", 4)) {
      data_found = true;
      std::vector<unsigned char> raw(sz);
      size_t got = fread(raw.data(), 1, sz, f);
      raw.resize(got);
      for (size_t i = 0; i + 1 < raw.size(); i += 2) {
        int16_t s;
        memcpy(&s, &raw[i], 2);
        w.pcm.push_back(s);
      }
    } else {
      fseek(f, sz + (sz & 1), SEEK_CUR);
    }
  }
  fclose(f);
  w.ok = fmt_found && data_found && bits == 16;
  if (w.ok && channels == 2) {  // naive stereo downmix
    std::vector<int16_t> mono(w.pcm.size() / 2);
    for (size_t i = 0; i < mono.size(); ++i)
      mono[i] = static_cast<int16_t>((int(w.pcm[2 * i]) + int(w.pcm[2 * i + 1])) / 2);
    w.pcm = std::move(mono);
  }
  return w;
}

// ---- mic source over one or more wav files ---------------------------------
class WavSource final : public gj::MicSource {
 public:
  WavSource(std::vector<Wav> wavs, bool realtime)
      : wavs_(std::move(wavs)), realtime_(realtime) {}

  std::size_t read(std::span<int16_t> dst) override {
    if (realtime_) {
      // pace one chunk per ~64 ms of audio, like the i2s dma cadence
      vTaskDelay(pdMS_TO_TICKS(64));
    }
    while (idx_ < wavs_.size()) {
      const Wav& w = wavs_[idx_];
      std::size_t avail = w.pcm.size() - pos_;
      if (avail == 0) { ++idx_; pos_ = 0; continue; }
      std::size_t n = dst.size() < avail ? dst.size() : avail;
      std::copy(w.pcm.begin() + pos_, w.pcm.begin() + pos_ + n, dst.begin());
      // pre-roll silence so the ring fills before the first window
      pos_ += n;
      return n;
    }
    return 0;
  }
  bool realtime() const override { return realtime_; }

 private:
  std::vector<Wav> wavs_;
  std::size_t idx_ = 0, pos_ = 0;
  bool realtime_;
};

// ---- csv alert sink ----------------------------------------------------------
struct CsvHooks {
  gj::PipelineHooks base;
  std::vector<std::string> rows;   // buffered; written once the scheduler ends
  long windows = 0;
  // last per-stage timings, filled by the stage hooks and read by the
  // detection hook (single consumer at write time; approximate by design —
  // exact percentiles come from the monitor task's mutex-guarded stats).
  std::atomic<int64_t> cap_us{0}, feat_us{0}, inf_us{0};
};

class ConsoleAlerts final : public gj::AlertSink {
 public:
  ConsoleAlerts() : start_(gj::Pipeline::micros()) {}
  void set_alarm(bool on) override {
    dprintf(STDERR_FILENO, "[alert] %s at t+%lldms\n", on ? "ALARM" : "clear",
            (long long)((gj::Pipeline::micros() - start_) / 1000));
  }
  void heartbeat(const gj::StageTiming&, float, bool) override {}

 private:
  int64_t start_;
};

// runs inside the detect task: never touch locked stdio here, only buffer.
void on_detection(void* user, const gj::Detection& d) {
  auto* h = static_cast<CsvHooks*>(user);
  char line[176];
  int n = snprintf(line, sizeof line, "%u,%ld,%.4f,%.1f,%d,%d,%lld,%lld,%lld\n",
                   d.hop, h->windows++, d.score, d.peak_dbfs, d.gated ? 1 : 0,
                   d.alarm ? 1 : 0, (long long)h->cap_us, (long long)h->feat_us,
                   (long long)h->inf_us);
  h->rows.emplace_back(line, static_cast<std::size_t>(n));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s <model.onnx> <audio.wav...> [--realtime] [--csv-out FILE] "
                    "[--threshold F]\n", argv[0]);
    return 2;
  }
  bool realtime = false;
  const char* csv_path = nullptr;
  float threshold = 0.5f;
  std::vector<std::string> wavs;
  for (int i = 2; i < argc; ++i) {
    if (!strcmp(argv[i], "--realtime")) realtime = true;
    else if (!strcmp(argv[i], "--csv-out") && i + 1 < argc) csv_path = argv[++i];
    else if (!strcmp(argv[i], "--threshold") && i + 1 < argc) threshold = atof(argv[++i]);
    else wavs.push_back(argv[i]);
  }

  FILE* model_f = fopen(argv[1], "rb");
  if (!model_f) { fprintf(stderr, "cannot open model %s\n", argv[1]); return 2; }
  fseek(model_f, 0, SEEK_END);
  long msz = ftell(model_f);
  fseek(model_f, 0, SEEK_SET);
  std::vector<std::byte> model(static_cast<std::size_t>(msz));
  if (fread(model.data(), 1, model.size(), model_f) != model.size()) { fprintf(stderr, "short read\n"); return 2; }
  fclose(model_f);

  std::vector<Wav> clips;
  for (auto& p : wavs) {
    Wav w = read_wav(p);
    if (!w.ok) { fprintf(stderr, "bad wav: %s\n", p.c_str()); return 2; }
    clips.push_back(std::move(w));
  }

  CsvHooks hooks;
  hooks.base.on_detection = on_detection;
  hooks.base.user = &hooks;
  hooks.base.on_capture = [](void* u, int64_t us) { static_cast<CsvHooks*>(u)->cap_us = us; };
  hooks.base.on_features = [](void* u, int64_t us) { static_cast<CsvHooks*>(u)->feat_us = us; };
  hooks.base.on_infer = [](void* u, int64_t us) { static_cast<CsvHooks*>(u)->inf_us = us; };

  ConsoleAlerts alerts;
  WavSource src(std::move(clips), realtime);

  // everything freertos-flavored (pipeline tasks, rtos objects) is created
  // from inside a task, after the scheduler is up — the posix port forbids
  // touching kernel state from a plain pthread like main().
  struct Boot {
    WavSource* src;
    ConsoleAlerts* alerts;
    std::vector<std::byte>* model;
    float threshold;
    CsvHooks* hooks;
  } boot{&src, &alerts, &model, threshold, &hooks};
  (void)csv_path;

  auto boot_fn = [](void* arg) {
    auto* b = static_cast<Boot*>(arg);
    static gj::Pipeline* pipe = nullptr;
    pipe = new gj::Pipeline(*b->src, *b->alerts, *b->model, b->threshold, b->hooks->base);
    if (!pipe->start()) { dprintf(STDERR_FILENO, "pipeline failed to start\n"); exit(1); }

    auto watcher = [](void* arg2) {
      auto* p = static_cast<gj::Pipeline*>(arg2);
      if (getenv("GJ_TRACE")) dprintf(STDERR_FILENO, "[watch] alive\n");
      while (!p->stopped()) vTaskDelay(pdMS_TO_TICKS(50));
      if (getenv("GJ_TRACE")) dprintf(STDERR_FILENO, "[watch] stop seen, draining\n");
      vTaskDelay(pdMS_TO_TICKS(250));  // let detect drain its queue
      vTaskEndScheduler();
      vTaskDelete(nullptr);
    };
    xTaskCreate(watcher, "watcher", 4096, pipe, 0, nullptr);
    vTaskDelete(nullptr);
  };
  xTaskCreate(boot_fn, "boot", 8192, &boot, gj::Pipeline::kCapturePrio + 1, nullptr);

  vTaskStartScheduler();   // parks main; returns after vTaskEndScheduler

  // main is a plain pthread again: safe to use locked stdio and write files.
  FILE* out = csv_path ? fopen(csv_path, "w") : stdout;
  fprintf(out, "hop,window,score,peak_dbfs,gated,alarm,capture_us,features_us,infer_us\n");
  for (auto& r : hooks.rows) fputs(r.c_str(), out);
  if (out != stdout) fclose(out);
  return 0;
}
