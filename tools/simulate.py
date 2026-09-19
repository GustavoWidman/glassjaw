"""the ponderada's "test code": simulate anomalies through the real freertos
pipeline and measure performance.

reconstructs each esc-50 test clip from the window cache, streams the whole
test set through build/sim/gj_sim in one run (exact firmware task graph), and
reports:

  - window-level accuracy/auc vs the dataset labels
  - clip-level alarm metrics (a clip counts as detected if any window alarms)
  - per-stage latency stats (capture/features/infer + p50/p95/p99)

usage:
  tools/run.sh tools/simulate.py [--model models/detector_int8.onnx]
                                 [--threshold 0.42] [--limit 200]
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).parent.parent
SIM = REPO / "build" / "sim" / "gj_sim"
HOP = 8000
WINDOWS_PER_CLIP = 9        # dataset windows per 5 s clip
STREAM_WINDOWS_PER_CLIP = 10  # the continuous stream yields one extra (straddle) window


def rebuild_clips() -> tuple[np.ndarray, np.ndarray]:
    d = np.load(REPO / "data" / "test.npz")
    x, y, cat = d["x"], d["y"], d["cat"]
    n_clips = len(x) // WINDOWS_PER_CLIP
    audio = []
    for c in range(n_clips):
        ws = x[c * WINDOWS_PER_CLIP:(c + 1) * WINDOWS_PER_CLIP]
        # windows overlap 50%: first half of w0 + second halves of the rest
        clip = np.concatenate([ws[0][:HOP]] + [w[HOP:] for w in ws])
        audio.append(clip)
    return np.stack(audio), y.reshape(n_clips, -1)[:, 0], cat.reshape(n_clips, -1)[:, 0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/detector_int8.onnx")
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--limit", type=int, default=0, help="only first N clips (quick run)")
    ap.add_argument("--out", default="docs/results/sim_report.json")
    args = ap.parse_args()

    audio, y, cat = rebuild_clips()
    if args.limit:
        audio, y, cat = audio[:args.limit], y[:args.limit], cat[:args.limit]

    with tempfile.TemporaryDirectory() as td:
        wavs = []
        for i, clip in enumerate(audio):
            p = Path(td) / f"{i:05d}.wav"
            import wave
            with wave.open(str(p), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes(clip.astype(np.int16).tobytes())
            wavs.append(str(p))

        csv_path = Path(td) / "out.csv"
        keep_csv = Path(args.out).with_suffix(".csv")
        cmd = [str(SIM), str(REPO / args.model), *wavs,
               "--csv-out", str(csv_path), "--threshold", str(args.threshold)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if r.returncode != 0:
            print(r.stderr[-2000:])
            sys.exit(1)
        rows = list(csv.DictReader(open(csv_path)))
        import shutil
        shutil.copy(csv_path, keep_csv)

    # window -> clip attribution, exactly as the pipeline sees it:
    # the concatenated stream is 80_000 samples per clip and the hop grid is
    # global, so stream window i starts at sample 8000*i and belongs to clip
    # floor(i/10). window 0 of each clip (i%10==0, i>0) straddles the clip
    # boundary (half previous clip) and is excluded from window metrics —
    # counting it would smear labels. clip-level metrics keep it: an alarm
    # caused by a straddling window is still a real alarm for that clip.
    scores = np.array([float(x["score"]) for x in rows])
    alarms = np.array([int(x["alarm"]) for x in rows])
    gated = np.array([int(x["gated"]) for x in rows])
    cap = np.array([int(x["capture_us"]) for x in rows])
    feat = np.array([int(x["features_us"]) for x in rows])
    inf = np.array([int(x["infer_us"]) for x in rows])

    # window k covers stream samples [8000k, 8000k+16000) (k=0 is the warmup
    # window). clips are 80000 samples back-to-back, so the window CENTER
    # (8000k+8000) identifies the clip whose audio dominates the window —
    # boundary windows straddle two clips and inherit the fresher half, which
    # matches detector semantics (newest evidence wins).
    hops = np.array([int(x["hop"]) for x in rows])
    # hop counts capture-side semaphore gives. give k fires when ~8000*k
    # samples have arrived, so window k covers ~[8000(k-1), 8000(k-1)+16000):
    # its center sits at ~8000*(k-1), inside clip floor((k-1)/10).
    # hop 1 is the warmup window (ring still short) -> excluded from metrics.
    warm = hops <= 1
    clip_of_window = np.minimum(np.maximum(hops - 2, 0) // STREAM_WINDOWS_PER_CLIP,
                                len(y) - 1)
    in_clip = (np.maximum(hops - 2, 0) % STREAM_WINDOWS_PER_CLIP) <= WINDOWS_PER_CLIP - 1
    expected = len(y) * WINDOWS_PER_CLIP - 1
    if len(hops) == 0 or (hops[-1] + 1) < expected - 2:
        print(f"FATAL: pipeline produced too few windows: last hop {hops[-1] if len(hops) else -1}, "
              f"expected ~{expected}")
        sys.exit(1)

    yw = y[clip_of_window].astype(int)          # per-window labels (clip label)
    audible = gated == 0                      # bool, not ~int (that flips bits)
    wa = (scores >= args.threshold).astype(int)
    report = {
        "model": args.model,
        "threshold": args.threshold,
        "windows": len(rows),
        "gated_windows": int(gated.sum()),
    }
    mk = audible & ~warm & in_clip
    if mk.sum() > 0 and len(np.unique(yw[mk])) > 1:
        ya = yw[mk]
        wa = wa[mk]
        sc = scores[mk]
        tp = int(((wa == 1) & (ya == 1)).sum())
        fp = int(((wa == 1) & (ya == 0)).sum())
        fn = int(((wa == 0) & (ya == 1)).sum())
        tn = int(((wa == 0) & (ya == 0)).sum())
        report["window_metrics"] = {
            "auc": float(roc_auc_score(ya, sc)),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": (tp + tn) / max(len(ya), 1),
        }
        # clip level: a clip is "detected" if any window alarms
        clip_alarm = np.zeros(len(y), dtype=int)
        np.maximum.at(clip_alarm, clip_of_window[in_clip], alarms[in_clip])
        report["clip_metrics"] = {
            "recall": float((clip_alarm[y == 1] == 1).mean()),
            "false_alarm_clips": int(((clip_alarm == 1) & (y == 0)).sum()),
            "normal_clips": int((y == 0).sum()),
        }
    report["latency_us"] = {
        "capture": {"p50": int(np.percentile(cap, 50)), "p95": int(np.percentile(cap, 95))},
        "features": {"p50": int(np.percentile(feat, 50)), "p95": int(np.percentile(feat, 95))},
        "infer": {"p50": int(np.percentile(inf, 50)), "p95": int(np.percentile(inf, 95))},
        "notes": "host sim (posix port) on this machine; esp32 numbers come from the device run",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
