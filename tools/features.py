"""DSP feature spec for glassjaw.

single source of truth for the feature pipeline. the C++ core in src/ mirrors
this module 1:1; tests/golden vectors pin the two implementations together.

window:      512 samples (32 ms) at 16 khz, hop 256 (16 ms)
spectrum:    512-point radix-2 fft, power = (re^2 + im^2) / n_fft
filterbank:  26 htk mel triangles, 50..8000 hz, peak 1.0, no area norm
cepstrum:    dct-ii orthonormal over log10 mel energies, c1..c13
frame feats: rms, zcr, spectral centroid, 85% spectral rolloff
window agg:  32 floats (see FEATURE_NAMES)
"""
from __future__ import annotations

import numpy as np

SR = 16_000
N_FFT = 512
HOP = 256
N_MELS = 26
F_MIN = 50.0
F_MAX = 8_000.0
N_MFCC = 13
ROLLOFF = 0.85
INFERENCE_WINDOW = 16_000          # 1.0 s of audio
EPS = 1e-10

FRAME_LEN = 1 + (INFERENCE_WINDOW - N_FFT) // HOP   # 61 frames

FEATURE_NAMES = (
    ["rms_mean", "rms_max", "zcr_mean", "centroid_mean", "centroid_std", "rolloff_mean"]
    + [f"mfcc{i}_mean" for i in range(1, N_MFCC + 1)]
    + [f"mfcc{i}_std" for i in range(1, N_MFCC + 1)]
)
N_FEATURES = len(FEATURE_NAMES)    # 32


def hamming(n: int = N_FFT) -> np.ndarray:
    """symmetric hamming window, identical formula in C++."""
    return 0.54 - 0.46 * np.cos(2.0 * np.pi * np.arange(n) / (n - 1))


def hz_to_mel(f: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_to_hz(m: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank() -> np.ndarray:
    """26 x 257 triangle weights. bins outside every triangle are zero."""
    n_bins = N_FFT // 2 + 1
    mel_pts = np.linspace(hz_to_mel(F_MIN), hz_to_mel(F_MAX), N_MELS + 2)
    bin_hz = np.arange(n_bins) * SR / N_FFT
    bin_mel = hz_to_mel(bin_hz)

    fb = np.zeros((N_MELS, n_bins))
    for j in range(N_MELS):
        left, center, right = mel_pts[j], mel_pts[j + 1], mel_pts[j + 2]
        up = (bin_mel - left) / (center - left)
        down = (right - bin_mel) / (right - center)
        fb[j] = np.clip(np.minimum(up, down), 0.0, None)
    return fb


_DCT_K: np.ndarray | None = None


def dct_matrix() -> np.ndarray:
    """orthonormal dct-ii matrix, 26 x 26. row 0 is c0 (dropped)."""
    global _DCT_K
    if _DCT_K is None:
        n = N_MELS
        k = np.arange(n)
        m = (2.0 * k[None, :] + 1.0) * k[:, None] * np.pi / (2.0 * n)
        m = np.cos(m)
        m[0] *= 1.0 / np.sqrt(2.0)
        m *= np.sqrt(2.0 / n)
        _DCT_K = m
    return _DCT_K


def frame_features(pcm: np.ndarray, fb: np.ndarray, win: np.ndarray, dct: np.ndarray) -> np.ndarray:
    """all frame-level features for one inference window.

    pcm: int16, len == INFERENCE_WINDOW. returns (FRAME_LEN, 4 + N_MFCC).
    """
    x = pcm.astype(np.float64) / 32768.0
    frames = np.lib.stride_tricks.sliding_window_view(x, N_FFT)[::HOP]
    frames = frames[:FRAME_LEN]

    w = win[None, :]
    spec = np.fft.rfft(frames * w, n=N_FFT, axis=1)
    power = (spec.real**2 + spec.imag**2) / N_FFT
    mag = np.sqrt(power)

    rms = np.sqrt(np.mean(frames**2, axis=1))
    zcr = np.mean((frames[:, :-1] * frames[:, 1:]) < 0.0, axis=1)

    freqs = np.arange(N_FFT // 2 + 1) * SR / N_FFT
    mag_sum = mag.sum(axis=1, keepdims=True) + EPS
    centroid = (mag @ freqs) / mag_sum[:, 0]

    cum = np.cumsum(mag, axis=1)
    rolloff_bin = np.argmax(cum >= ROLLOFF * cum[:, -1:], axis=1)
    rolloff = freqs[rolloff_bin]

    mel_power = power @ fb.T
    log_mel = np.log10(mel_power + EPS)
    mfcc = (log_mel @ dct.T)[:, 1 : N_MFCC + 1]   # drop c0, keep c1..c13

    return np.column_stack([rms, zcr, centroid, rolloff, mfcc])


def extract(pcm: np.ndarray) -> np.ndarray:
    """int16 16 khz mono -> 31-float feature vector (float32)."""
    assert pcm.dtype == np.int16 and len(pcm) == INFERENCE_WINDOW
    fb = mel_filterbank()
    ff = frame_features(pcm, fb, hamming(), dct_matrix())

    rms, zcr, centroid, rolloff, mfcc = np.split(ff, [1, 2, 3, 4], axis=1)
    vec = np.concatenate(
        [
            rms.mean(axis=0),       # rms_mean (1)
            rms.max(axis=0),        # rms_max (1)
            zcr.mean(axis=0),       # zcr_mean (1)
            centroid.mean(axis=0),  # centroid_mean (1)
            centroid.std(axis=0),   # centroid_std (1)
            rolloff.mean(axis=0),   # rolloff_mean (1)
            mfcc.mean(axis=0),      # mfcc mean (13)
            mfcc.std(axis=0),       # mfcc std (13)
        ]
    )
    return vec.astype(np.float32)


if __name__ == "__main__":
    import json
    import sys

    rng = np.random.default_rng(42)
    pcm = (rng.standard_normal(INFERENCE_WINDOW) * 3000).astype(np.int16)
    fv = extract(pcm)
    frames = frame_features(pcm, mel_filterbank(), hamming(), dct_matrix())
    json.dump(
        {
            "pcm": pcm.tolist(),
            "features": fv.tolist(),
            "frame_features": frames.tolist(),
        },
        open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/golden_features.json", "w"),
    )
    print(f"frames={frames.shape} features={fv.shape}")


def log_mel(pcm: np.ndarray) -> np.ndarray:
    """int16 16k window -> log10 mel spectrogram, band-major (kMelBands, 61).

    identical layout to gj::log_mel_spectrogram in src/dsp.cpp:
    spec[band, frame] = log10(sum_bin fb[band,bin] * power[bin] + eps).
    """
    assert pcm.dtype == np.int16 and len(pcm) == INFERENCE_WINDOW
    fb = mel_filterbank()
    win = hamming()
    x = pcm.astype(np.float64) / 32768.0
    frames = np.lib.stride_tricks.sliding_window_view(x, N_FFT)[::HOP][:FRAME_LEN]
    spec = np.fft.rfft(frames * win[None, :], n=N_FFT, axis=1)
    power = (spec.real**2 + spec.imag**2) / N_FFT
    return np.log10(power @ fb.T + EPS).T          # (bands, frames)


def peak_dbfs(pcm: np.ndarray) -> float:
    peak = int(np.max(np.abs(pcm.astype(np.int32))))
    if peak == 0:
        return -240.0
    return float(20.0 * np.log10(peak / 32768.0))
