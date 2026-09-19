"""regenerate the frame/mel/peak caches under data/cache from data/*.npz.

the caches are derived artifacts (gitignored) — rebuild them after
prepare_dataset.py so train.py and simulate.py don't recompute 10k windows.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from features import N_FFT, HOP, SR, EPS, hamming, mel_filterbank


def frame_feats(pcm, fb, win):
    x = pcm.astype(np.float64) / 32768.0
    fr = np.lib.stride_tricks.sliding_window_view(x, N_FFT)[::HOP]
    fr = fr[: 1 + (len(x) - N_FFT) // HOP]
    spec = np.fft.rfft(fr * win[None, :], n=N_FFT, axis=1)
    power = (spec.real**2 + spec.imag**2) / N_FFT
    return np.log10(power @ fb.T + EPS)


fb, win = mel_filterbank(), hamming()
out = Path("data/cache")
out.mkdir(parents=True, exist_ok=True)
for split in ["train", "test"]:
    d = np.load(f"data/{split}.npz")
    np.save(out / f"mels_{split}.npy",
            np.stack([frame_feats(w, fb, win) for w in d["x"]]).astype(np.float32))
    peak = np.array([20 * np.log10(np.abs(w.astype(np.float32)).max() / 32768 + 1e-12)
                     for w in d["x"]])
    np.save(out / f"peak_{split}.npy", peak.astype(np.float32))
    print(split, len(d["x"]), "windows")
