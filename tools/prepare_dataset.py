"""esc-50 -> glassjaw dataset cache.

resamples every selected clip to 16 khz mono int16 and caches it as a single
npz so train.py / simulate.py never touch the 44.1 khz wavs again.

class plan (break-in detector):
  anomaly : glass_breaking, siren
  normal  : everyday household + ambient sounds that should NOT trip the alarm

split by esc-50 fold: folds 1..4 train, fold 5 test. the fold column is the
dataset's own speaker/recording-disjoint split, so no clip leaks across.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).parent))
from features import INFERENCE_WINDOW, SR  # noqa: E402

ANOMALY_CLASSES = ["glass_breaking", "siren"]

NORMAL_CLASSES = [
    "breathing", "coughing", "footsteps", "laughing", "snoring",
    "brushing_teeth", "drinking_sipping", "crying_and_sobbing",
    "door_wood_knock", "door_wood_creaks", "mouse_click", "keyboard_typing",
    "can_opening", "washing_machine", "vacuum_cleaner", "clock_alarm",
    "clock_tick", "toilet_flush", "pouring_water", "dog", "cat",
    "wind", "rain", "crackling_fire", "crickets", "chirping_birds",
    "water_drops",
]

TEST_FOLD = 5


def load_split(esc_root: Path, split: str):
    import csv

    rows = list(csv.DictReader(open(esc_root / "meta" / "esc50.csv")))
    clips = []
    for r in rows:
        if r["category"] in ANOMALY_CLASSES:
            label = 1
        elif r["category"] in NORMAL_CLASSES:
            label = 0
        else:
            continue
        fold = int(r["fold"])
        if split == "train" and fold == TEST_FOLD:
            continue
        if split == "test" and fold != TEST_FOLD:
            continue
        clips.append((r["filename"], label, r["category"], fold))
    return clips


def resample(path: Path) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float64", always_2d=True)
    mono = wav.mean(axis=1)
    if sr != SR:
        g = np.gcd(SR, sr)
        mono = resample_poly(mono, SR // g, sr // g)
    peak = np.max(np.abs(mono))
    if peak > 1.0:
        mono = mono / peak
    return (np.clip(mono, -1.0, 1.0) * 32767.0).astype(np.int16)


def windows(pcm: np.ndarray):
    """1 s windows, 0.5 s hop. drops the last partial window."""
    step = INFERENCE_WINDOW // 2
    n = 0
    while n + INFERENCE_WINDOW <= len(pcm):
        yield pcm[n : n + INFERENCE_WINDOW]
        n += step


def main(esc_root: Path, out: Path):
    for split in ("train", "test"):
        clips = load_split(esc_root, split)
        xs, ys, meta = [], [], []
        for fname, label, cat, fold in clips:
            pcm = resample(esc_root / "audio" / fname)
            for w in windows(pcm):
                xs.append(w)
                ys.append(label)
                meta.append((fname, cat, fold))
        np.savez_compressed(
            out / f"{split}.npz",
            x=np.stack(xs),
            y=np.array(ys, dtype=np.int8),
            fname=np.array([m[0] for m in meta]),
            cat=np.array([m[1] for m in meta]),
            fold=np.array([m[2] for m in meta], dtype=np.int8),
        )
        pos = int(np.sum(ys))
        print(f"{split}: {len(ys)} windows ({pos} anomaly / {len(ys)-pos} normal) from {len(clips)} clips")


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/esc50/ESC-50-master")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data")
    out.mkdir(exist_ok=True)
    main(root, out)
