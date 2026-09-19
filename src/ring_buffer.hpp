// single-producer/single-consumer sample ring.
// capture writes, features reads: power-of-two ring, no lock needed.
#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <span>

namespace gj {

template <typename T, std::size_t CapacityPow2>
class SpscRing {
  static_assert((CapacityPow2 & (CapacityPow2 - 1)) == 0, "capacity must be a power of two");

 public:
  // copy n samples in; on overflow the oldest data is dropped.
  std::size_t write(std::span<const T> src) {
    const std::size_t n = src.size();
    const std::size_t cap = CapacityPow2;
    std::size_t head = head_.load(std::memory_order_relaxed);
    std::size_t tail = tail_.load(std::memory_order_acquire);
    std::size_t used = head - tail;
    if (n > cap) {
      src = src.last(cap);  // keep only the newest slice that fits
    }
    const std::size_t free_space = cap - used;
    const std::size_t count = src.size();
    if (count > free_space) {
      tail += count - free_space;               // drop oldest
      tail_.store(tail, std::memory_order_release);
    }
    for (std::size_t i = 0; i < count; ++i) {
      buf_[(head + i) & (cap - 1)] = src[i];
    }
    head_.store(head + count, std::memory_order_release);
    return count;
  }

  // copy up to dst.size() samples out; returns how many were read.
  std::size_t read(std::span<T> dst) {
    const std::size_t cap = CapacityPow2;
    std::size_t head = head_.load(std::memory_order_acquire);
    std::size_t tail = tail_.load(std::memory_order_relaxed);
    const std::size_t available = head - tail;
    const std::size_t count = dst.size() < available ? dst.size() : available;
    for (std::size_t i = 0; i < count; ++i) dst[i] = buf_[(tail + i) & (cap - 1)];
    tail_.store(tail + count, std::memory_order_release);
    return count;
  }

  // snapshot without consuming: newest `dst.size()` samples (zero-padded if not
  // yet full). the feature task uses this for its 50%-overlap sliding window.
  void peek_newest(std::span<T> dst) const {
    const std::size_t cap = CapacityPow2;
    std::size_t head = head_.load(std::memory_order_acquire);
    std::size_t tail = tail_.load(std::memory_order_acquire);
    const std::size_t available = head - tail;
    const std::size_t n = dst.size();
    const std::size_t take = available < n ? available : n;
    for (std::size_t i = 0; i < n; ++i) {
      dst[i] = static_cast<T>(0);
    }
    for (std::size_t i = 0; i < take; ++i) {
      dst[n - take + i] = buf_[(head - take + i) & (cap - 1)];
    }
  }

  std::size_t available() const {
    return head_.load(std::memory_order_acquire) - tail_.load(std::memory_order_acquire);
  }

 private:
  std::array<T, CapacityPow2> buf_{};
  alignas(64) std::atomic<std::size_t> head_{0};
  alignas(64) std::atomic<std::size_t> tail_{0};
};

}  // namespace gj
