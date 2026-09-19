// protobuf wire-format reader, enough for onnx model files. unknown
// fields are skipped by wire type.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace gj::pb {

class Reader {
 public:
  Reader(std::span<const std::byte> data, std::size_t pos = 0)
      : data_(data), pos_(pos) {}

  bool eof() const { return pos_ >= data_.size(); }
  std::size_t pos() const { return pos_; }
  std::span<const std::byte> rest() const { return data_.subspan(pos_); }

  uint64_t varint() {
    uint64_t v = 0;
    int shift = 0;
    for (;;) {
      if (pos_ >= data_.size()) return v;  // truncated input: caller checks eof
      std::byte b = data_[pos_++];
      v |= static_cast<uint64_t>(b & std::byte{0x7f}) << shift;
      if ((b & std::byte{0x80}) == std::byte{0}) return v;
      shift += 7;
      if (shift > 63) return v;
    }
  }

  // returns {field_number, wire_type} packed; wire_type in low 3 bits.
  uint32_t tag() { return static_cast<uint32_t>(varint()); }

  std::span<const std::byte> bytes() {
    uint64_t len = varint();
    len = len > remaining() ? remaining() : len;
    auto out = data_.subspan(pos_, len);
    pos_ += len;
    return out;
  }

  uint32_t fixed32() {
    if (remaining() < 4) { pos_ = data_.size(); return 0; }
    uint32_t v;
    auto* p = reinterpret_cast<const unsigned char*>(data_.data() + pos_);
    v = static_cast<uint32_t>(p[0]) | static_cast<uint32_t>(p[1]) << 8 |
        static_cast<uint32_t>(p[2]) << 16 | static_cast<uint32_t>(p[3]) << 24;
    pos_ += 4;
    return v;
  }

  uint64_t fixed64() {
    uint64_t lo = fixed32();
    uint64_t hi = fixed32();
    return lo | hi << 32;
  }

  void skip(uint32_t wire_type) {
    switch (wire_type) {
      case 0: varint(); break;
      case 1: pos_ += 8 >= remaining() ? remaining() : 8; break;
      case 2: bytes(); break;
      case 5: pos_ += 4 >= remaining() ? remaining() : 4; break;
      default: pos_ = data_.size(); break;  // groups unsupported: bail out
    }
  }

  std::size_t remaining() const { return data_.size() - pos_; }

 private:
  std::span<const std::byte> data_;
  std::size_t pos_;
};

inline std::string_view as_str(std::span<const std::byte> b) {
  return std::string_view(reinterpret_cast<const char*>(b.data()), b.size());
}

}  // namespace gj::pb
