"""generate docs/demo clips for the in-class checkpoint.

synthetic clips (no esc-50 redistribution concerns): a glass-like burst
(broadband crack + high-frequency ring-out) and a quiet-room baseline.
rehearse the demo by feeding them straight to gj_sim, or play them at the
device microphone.
"""
import sys
from pathlib import Path
import wave

import numpy as np

SR = 16_000


def glass_like_burst(rng):
    t = np.arange(SR * 3) / SR
    x = np.zeros(SR * 3, np.float32)
    # initial crack: impulsive noise burst, fast decay
    crack_env = np.exp(-np.arange(SR // 4) / (SR * 0.004))
    x[: SR // 4] += rng.standard_normal(SR // 4) * crack_env * 12000
    # shards: sprinkle short decaying pings 100-800 ms after
    for _ in range(24):
        start = rng.integers(SR // 10, SR * 8 // 10)
        dur = rng.integers(120, 420)
        freq = rng.uniform(2800, 7200)
        env = np.exp(-np.arange(dur) / (SR * 0.01))
        x[start:start + dur] += np.sin(2 * np.pi * freq * np.arange(dur) / SR) * env * rng.uniform(500, 3500)
    # faint room noise
    x += rng.standard_normal(SR * 3) * 60
    return x


def quiet_room(rng):
    t = np.arange(SR * 3) / SR
    x = rng.standard_normal(SR * 3) * 120  # hvac-ish noise floor
    x += (np.sin(2 * np.pi * 120 * t) * 40).astype(np.float32)
    return x


def write(path, pcm):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(np.clip(pcm, -32768, 32767).astype(np.int16).tobytes())


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs/demo")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    write(out / "glass_burst.wav", glass_like_burst(rng))
    write(out / "quiet_room.wav", quiet_room(rng))
    print(f"wrote {out}/glass_burst.wav and quiet_room.wav")
