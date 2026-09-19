#include "doctest.h"

#include <vector>

#include "../src/ring_buffer.hpp"

TEST_CASE("spsc ring: basic write/read roundtrip") {
  gj::SpscRing<int16_t, 8> ring;
  std::vector<int16_t> in{1, 2, 3};
  CHECK(ring.write(in) == 3);
  CHECK(ring.available() == 3);
  std::vector<int16_t> out(3);
  CHECK(ring.read(out) == 3);
  CHECK(out == std::vector<int16_t>{1, 2, 3});
  CHECK(ring.available() == 0);
}

TEST_CASE("spsc ring: overflow drops oldest, keeps newest") {
  gj::SpscRing<int16_t, 4> ring;
  std::vector<int16_t> in{1, 2, 3, 4, 5, 6};
  CHECK(ring.write(in) == 4);  // accepts a full ring, 1,2 dropped
  std::vector<int16_t> out(4);
  CHECK(ring.read(out) == 4);
  CHECK(out == std::vector<int16_t>{3, 4, 5, 6});
}

TEST_CASE("spsc ring: peek_newest zero-pads short history") {
  gj::SpscRing<int16_t, 16> ring;
  std::vector<int16_t> in{7, 8};
  ring.write(in);
  std::vector<int16_t> out(4);
  ring.peek_newest(out);
  CHECK(out == std::vector<int16_t>{0, 0, 7, 8});
}

TEST_CASE("spsc ring: peek_newest full window, newest last") {
  gj::SpscRing<int16_t, 8> ring;
  for (int i = 0; i < 12; ++i) {
    std::vector<int16_t> one{static_cast<int16_t>(i)};
    ring.write(one);
  }
  std::vector<int16_t> out(6);
  ring.peek_newest(out);  // history holds 6,7,8,9,10,11
  CHECK(out == std::vector<int16_t>{6, 7, 8, 9, 10, 11});
}

TEST_CASE("spsc ring: sliding window via peek keeps consume-free overlap") {
  gj::SpscRing<int16_t, 16> ring;
  std::vector<int16_t> a{0, 0, 0, 0};
  ring.write(a);
  std::vector<int16_t> w1(4);
  ring.peek_newest(w1);
  CHECK(w1 == a);
  std::vector<int16_t> b{5, 6};
  ring.write(b);  // window now 0,0,5,6 (b starts at t=4 in stream terms)
  std::vector<int16_t> w2(4);
  ring.peek_newest(w2);
  CHECK(w2 == std::vector<int16_t>{0, 0, 5, 6});
}
