// alarm logic: threshold + debounce. two audible hits arm the alarm,
// three misses clear it. gated (inaudible) windows never arm it but do
// count toward clearing.
#pragma once

#include "pipeline.hpp"

namespace gj {

class Detector {
 public:
  explicit Detector(float threshold) : threshold_(threshold) {}

  // feed one inference result; returns the (possibly unchanged) alarm state.
  bool update(const Detection& d) {
    const bool above = !d.gated && d.score >= threshold_;
    if (above) {
      ++above_run_;
      below_run_ = 0;
    } else {
      ++below_run_;
      above_run_ = 0;
    }
    if (!alarm_ && above_run_ >= kAlarmHold) alarm_ = true;
    if (alarm_ && below_run_ >= kClearHold) alarm_ = false;
    return alarm_;
  }

  bool alarm() const { return alarm_; }
  float threshold() const { return threshold_; }

 private:
  float threshold_;
  int above_run_ = 0;
  int below_run_ = 0;
  bool alarm_ = false;
};

}  // namespace gj
