// latency stats: fixed-capacity ring, percentiles, no allocation.
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

namespace gj {

template <std::size_t N>
class LatencyStats {
 public:
  void push(int64_t micros) {
    if (count_ < N) {
      buf_[count_++] = micros;
    } else {
      buf_[idx_ % N] = micros;
      ++idx_;
    }
  }

  std::size_t count() const { return count_; }

  // p in [0,1]. linear interpolation between order statistics.
  int64_t percentile(float p) const {
    if (count_ == 0) return 0;
    std::array<int64_t, N> tmp{};
    std::copy(buf_.begin(), buf_.begin() + count_, tmp.begin());
    std::sort(tmp.begin(), tmp.begin() + count_);
    float pos = p * static_cast<float>(count_ - 1);
    std::size_t lo = static_cast<std::size_t>(pos);
    std::size_t hi = lo + 1 < count_ ? lo + 1 : lo;
    float frac = pos - static_cast<float>(lo);
    return static_cast<int64_t>(tmp[lo] + (tmp[hi] - tmp[lo]) * frac);
  }

  int64_t mean() const {
    if (count_ == 0) return 0;
    int64_t sum = 0;
    for (std::size_t i = 0; i < count_; ++i) sum += buf_[i];
    return sum / static_cast<int64_t>(count_);
  }

  int64_t max() const {
    if (count_ == 0) return 0;
    int64_t m = buf_[0];
    for (std::size_t i = 1; i < count_; ++i) m = std::max(m, buf_[i]);
    return m;
  }

  void reset() { count_ = 0; idx_ = 0; }

 private:
  std::array<int64_t, N> buf_{};
  std::size_t count_ = 0;
  std::size_t idx_ = 0;
};

}  // namespace gj
