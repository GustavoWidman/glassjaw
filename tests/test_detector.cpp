#include "doctest.h"

#include "../src/detector.hpp"

using gj::Detection;

namespace {
Detection mk(float score, bool gated = false) { return Detection{score, -10.0f, gated, false, 0}; }
}  // namespace

TEST_CASE("detector: needs kAlarmHold consecutive hits to latch") {
  gj::Detector d(0.5f);
  CHECK_FALSE(d.update(mk(0.9f)));
  CHECK(d.update(mk(0.9f)));  // second consecutive hit latches
  CHECK(d.alarm());
}

TEST_CASE("detector: single dip does not clear") {
  gj::Detector d(0.5f);
  d.update(mk(0.9f));
  d.update(mk(0.9f));
  CHECK(d.alarm());
  d.update(mk(0.1f));
  CHECK(d.alarm());  // one miss is not enough
  d.update(mk(0.1f));
  d.update(mk(0.1f));
  CHECK_FALSE(d.alarm());  // kClearHold reached
}

TEST_CASE("detector: gated windows never raise, but count toward clear") {
  gj::Detector d(0.5f);
  d.update(mk(0.9f));
  CHECK(d.update(mk(0.9f)));
  d.update(mk(0.0f, /*gated=*/true));
  d.update(mk(0.0f, /*gated=*/true));
  CHECK(d.update(mk(0.0f, /*gated=*/true)) == false);
}

TEST_CASE("detector: re-arms after clearing") {
  gj::Detector d(0.5f);
  d.update(mk(0.9f));
  d.update(mk(0.9f));
  for (int i = 0; i < 3; ++i) d.update(mk(0.1f));
  CHECK_FALSE(d.alarm());
  CHECK_FALSE(d.update(mk(0.9f)));   // re-arm needs the full hold again
  CHECK(d.update(mk(0.9f)));
}
