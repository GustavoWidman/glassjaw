#include "doctest.h"

#include "../src/stats.hpp"

TEST_CASE("stats: percentiles on a small sample") {
  gj::LatencyStats<16> s;
  for (int64_t v : {10, 20, 30, 40}) s.push(v);
  CHECK(s.count() == 4);
  CHECK(s.percentile(0.0f) == 10);
  CHECK(s.percentile(1.0f) == 40);
  CHECK(s.percentile(0.5f) == 25);  // interpolated median
  CHECK(s.mean() == 25);
  CHECK(s.max() == 40);
}

TEST_CASE("stats: empty is safe") {
  gj::LatencyStats<8> s;
  CHECK(s.percentile(0.99f) == 0);
  CHECK(s.mean() == 0);
}

TEST_CASE("stats: wraps, keeps newest N") {
  gj::LatencyStats<4> s;
  for (int64_t v = 1; v <= 10; ++v) s.push(v);
  CHECK(s.count() == 4);
  CHECK(s.max() == 10);
  CHECK(s.mean() == (7 + 8 + 9 + 10) / 4);
}
