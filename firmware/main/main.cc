// glassjaw firmware — esp32 + inmp441 + led + buzzer.
//
// wiring (see docs/diagrams/wiring.svg):
//   inmp441 vdd -> 3v3, gnd -> gnd, sck -> gpio14, ws -> gpio15,
//           sd -> gpio32, l/r -> gnd
//   status led  -> gpio2  (onboard)
//   alarm  led  -> gpio25 (through 330r)
//   buzzer      -> gpio26 (active module)
//
// the pipeline itself (tasks, queues, onnx engine) is the exact same code
// the host simulator runs: src/app/pipeline.{hpp,cpp}.
#include <cstdio>
#include <cstring>

#include "driver/gpio.h"
#include "driver/i2s_std.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "app/pipeline.hpp"
#include "esp_heap_caps.h"

#include "config.hpp"

namespace {

constexpr gpio_num_t kPinI2sSck = GPIO_NUM_14;
constexpr gpio_num_t kPinI2sWs = GPIO_NUM_15;
constexpr gpio_num_t kPinI2sSd = GPIO_NUM_32;
constexpr gpio_num_t kPinStatusLed = GPIO_NUM_2;
constexpr gpio_num_t kPinAlarmLed = GPIO_NUM_5;    // D5 on the right header
constexpr gpio_num_t kPinAlarmLed2 = GPIO_NUM_25;  // also driven: 30-pin variant
// the ponderada lists the buzzer as optional ("LED + Buzzer (opcional)");
// this kit has none, so it is a compile-time option. alarm = fast-blinking
// led, which reads better on camera anyway.
#ifdef CONFIG_GLASSJAW_BUZZER
constexpr gpio_num_t kPinBuzzer = GPIO_NUM_26;
#endif

// the int8 model is embedded at build time (see main/CMakeLists.txt EMBED_FILES)
extern const uint8_t model_start[] asm("_binary_detector_int8_onnx_start");
extern const uint8_t model_end[] asm("_binary_detector_int8_onnx_end");

// inmp441 delivers 24-bit samples left-justified in a 32-bit frame. the
// shift sets the input gain: >>16 leaves ~12 db of headroom so a loud event
// next to the mic does not clip (at >>14 a phone speaker at arm's length
// saturates and the spectrogram loses the event signature entirely).
constexpr int kI2sShift = 16;

class I2sMicSource final : public gj::MicSource {
 public:
  bool init() {
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    chan_cfg.dma_desc_num = 8;
    chan_cfg.dma_frame_num = 256;
    esp_err_t err = i2s_new_channel(&chan_cfg, nullptr, &rx_);
    if (err != ESP_OK) return false;

    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(16000),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = kPinI2sSck,
            .ws = kPinI2sWs,
            .dout = I2S_GPIO_UNUSED,
            .din = kPinI2sSd,
            .invert_flags = {},
        },
    };
    // l/r tied to gnd: data arrives on the left slot. 32-bit frame with the
    // 24-bit sample msb-aligned; >> kI2sShift brings it to int16 range.
    std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;
    if (i2s_channel_init_std_mode(rx_, &std_cfg) != ESP_OK) return false;
    return i2s_channel_enable(rx_) == ESP_OK;
  }

  std::size_t read(std::span<int16_t> dst) override {
    // 4 kb dma buffer as a member-static: it must not live on the task stack
    // (the first firmware run overflowed capture exactly here).
    static std::array<int32_t, gj::Pipeline::kChunk> raw{};
    std::size_t bytes = 0;
    if (i2s_channel_read(rx_, raw.data(), raw.size() * sizeof(int32_t), &bytes,
                         portMAX_DELAY) != ESP_OK) {
      return 0;
    }
    std::size_t n = bytes / sizeof(int32_t);
    for (std::size_t i = 0; i < n && i < dst.size(); ++i) {
      int32_t s = raw[i] >> kI2sShift;
      dst[i] = static_cast<int16_t>(s < -32768 ? -32768 : (s > 32767 ? 32767 : s));
    }
    return n < dst.size() ? n : dst.size();
  }

  bool realtime() const override { return true; }

 private:
  i2s_chan_handle_t rx_ = nullptr;
};

class GpioAlerts final : public gj::AlertSink {
 public:
  void init() {
#ifdef CONFIG_GLASSJAW_BUZZER
    for (auto pin : {kPinStatusLed, kPinAlarmLed, kPinAlarmLed2, kPinBuzzer}) {
#else
    for (auto pin : {kPinStatusLed, kPinAlarmLed, kPinAlarmLed2}) {
#endif
      gpio_reset_pin(pin);
      gpio_set_direction(pin, GPIO_MODE_OUTPUT);
      gpio_set_level(pin, 0);
    }
  }

  void set_alarm(bool on) override {
    alarm_ = on;
    gpio_set_level(kPinAlarmLed, on ? 1 : 0);
    gpio_set_level(kPinAlarmLed2, on ? 1 : 0);
#ifdef CONFIG_GLASSJAW_BUZZER
    gpio_set_level(kPinBuzzer, on ? 1 : 0);
#endif
    printf("[alert] %s\n", on ? "GLASS BREAK ALARM" : "clear");
  }

  // the alarm led does double duty: 1 hz blink = alive, solid = alarm


  void tick(bool alarm) override {
    // no user-controllable onboard led on this board variant (gpio2 has
    // none; the red one near the left edge is the power indicator). the
    // external red led carries both signals in patterns that cannot be
    // confused: a 90 ms blip every ~2 s = alive, solid = alarm.
    if (alarm) return;  // solid, set_alarm drove it high
    static int n = 0;
    if (n % 32 == 0) {          // every 32 reads (~2 s): blip
      gpio_set_level(kPinAlarmLed, 1);
    } else if (n % 32 == 2) {   // ~130 ms later: off again
      gpio_set_level(kPinAlarmLed, 0);
    }
    ++n;
  }

  void heartbeat(const gj::StageTiming&, float score, bool alarm) override {
    // tuning telemetry: every window, score + peak level
    printf("[detect] score=%.3f alarm=%d\n", score, alarm ? 1 : 0);
  }

 private:
  bool alarm_ = false;
};

}  // namespace

extern "C" void app_main() {
  static I2sMicSource mic;
  static GpioAlerts alerts;
  alerts.init();
  if (!mic.init()) {
    printf("fatal: i2s init failed — check wiring (sck=%d ws=%d sd=%d)\n",
           kPinI2sSck, kPinI2sWs, kPinI2sSd);
    return;
  }

  auto model = std::span<const std::byte>(
      reinterpret_cast<const std::byte*>(model_start),
      static_cast<std::size_t>(model_end - model_start));

  gj::PipelineHooks hooks;
  hooks.on_detection = [](void*, const gj::Detection& d) {
    printf("[window] score=%.3f peak=%.0f gated=%d alarm=%d\n",
           d.score, d.peak_dbfs, d.gated ? 1 : 0, d.alarm ? 1 : 0);
  };
  hooks.on_infer = [](void*, int64_t us) {
    static int64_t acc = 0; static int n = 0;
    acc += us;
    if (++n % 60 == 0) printf("[infer] n=%d mean=%lld us\n", n, (long long)(acc / n));
  };
  static gj::Pipeline* pipe = new gj::Pipeline(mic, alerts, model, GLASSJAW_THRESHOLD, hooks);

  auto boot = [](void*) {
    printf("glassjaw: led = blip every 2s (alive) / solid (alarm), v4\n");
    for (int i = 0; i < 5; ++i) {
      gpio_set_level(kPinAlarmLed, 1);
      gpio_set_level(kPinAlarmLed2, 1);
      vTaskDelay(pdMS_TO_TICKS(90));
      gpio_set_level(kPinAlarmLed, 0);
      gpio_set_level(kPinAlarmLed2, 0);
      vTaskDelay(pdMS_TO_TICKS(90));
    }
    if (!pipe->start(/*core=*/1)) {
      printf("fatal: pipeline failed to start (model load?)\n");
    }
    vTaskDelete(nullptr);
  };
  xTaskCreate(boot, "boot", 8192, nullptr, gj::Pipeline::kCapturePrio + 1, nullptr);
}
