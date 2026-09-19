"""numpy cnn core shared by the trainer and experiments. im2col convolutions
with analytic gradients; verified against numerical differentiation."""
from __future__ import annotations

import numpy as np


def im2col(x, k, sh):
    B, H, W, C = x.shape
    oh, ow = (H - k) // sh + 1, (W - k) // sh + 1
    cols = np.empty((B, oh, ow, k * k * C), dtype=x.dtype)
    for i in range(k):
        for j in range(k):
            for c in range(C):
                cols[:, :, :, (i * k + j) * C + c] = x[:, i:i + sh * oh:sh, j:j + sh * ow:sh, c]
    return cols, (B, oh, ow)


def maxpool(x, k):
    B, H, W, C = x.shape
    oh, ow = H // k, W // k
    return x[:, :oh * k, :ow * k, :].reshape(B, oh, k, ow, k, C).max(axis=(2, 4))


def maxpool_bwd(x, dout, k):
    B, H, W, C = x.shape
    oh, ow = dout.shape[1], dout.shape[2]
    dx = np.zeros_like(x)
    xr = x[:, :oh * k, :ow * k, :].reshape(B, oh, k, ow, k, C)
    mx = xr.max(axis=(2, 4), keepdims=True)
    m = (xr == mx).astype(np.float32)
    m /= m.sum(axis=(2, 4), keepdims=True)
    dx[:, :oh * k, :ow * k, :] = (
        m * dout.repeat(k, 2).repeat(k, 3).reshape(B, oh, k, ow, k, C)
    ).reshape(B, oh * k, ow * k, C)
    return dx


def sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class Conv:
    def __init__(self, cin, cout, k, rng):
        self.w = (rng.standard_normal((k * k * cin, cout)) * np.sqrt(2.0 / (k * k * cin))).astype(np.float32)
        self.b = np.zeros(cout, np.float32)
        self.k, self.cin, self.cout = k, cin, cout

    def fwd(self, x):
        cols, (B, oh, ow) = im2col(x, self.k, 1)
        self.x, self.cols, self.oh, self.ow = x, cols, oh, ow
        return (cols.reshape(-1, cols.shape[-1]) @ self.w + self.b).reshape(B, oh, ow, self.cout)

    def bwd(self, dout):
        dfl = dout.reshape(-1, self.cout)
        self.gw = self.cols.reshape(-1, self.cols.shape[-1]).T @ dfl
        self.gb = dfl.sum(0)
        dcols = (dfl @ self.w.T).reshape(self.cols.shape)
        dx = np.zeros_like(self.x)
        k = self.k
        for i in range(k):
            for j in range(k):
                for c in range(self.cin):
                    dx[:, i:i + self.oh, j:j + self.ow, c] += dcols[:, :, :, (i * k + j) * self.cin + c]
        return dx


class Net:
    """conv(1->16) pool conv(16->32) pool conv(32->32) gap fc(1)."""

    def __init__(self, rng):
        self.c1 = Conv(1, 16, 3, rng)
        self.c2 = Conv(16, 32, 3, rng)
        self.c3 = Conv(32, 32, 3, rng)
        self.fc = (rng.standard_normal((32, 1)) * 0.1).astype(np.float32)
        self.fcb = np.zeros(1, np.float32)
        self.params = [self.c1.w, self.c1.b, self.c2.w, self.c2.b,
                       self.c3.w, self.c3.b, self.fc, self.fcb]
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]

    def forward(self, x):
        h1 = np.maximum(self.c1.fwd(x), 0)
        p1 = maxpool(h1, 2)
        h2 = np.maximum(self.c2.fwd(p1), 0)
        p2 = maxpool(h2, 2)
        h3 = np.maximum(self.c3.fwd(p2), 0)
        self.chain = (h1, p1, h2, p2, h3)
        return h3.mean(axis=(1, 2)) @ self.fc + self.fcb

    def backward(self, logit, yb, pw=1.0):
        h1, p1, h2, p2, h3 = self.chain
        p = sig(logit)
        flat = (p - yb[:, None]) * np.where(yb[:, None] > 0.5, pw, 1.0) / len(yb)
        gap = h3.mean(axis=(1, 2))
        dh3 = (flat @ self.fc.T)[:, None, None, :] / (h3.shape[1] * h3.shape[2])
        dh3 = dh3 * (h3 > 0)
        dp2 = self.c3.bwd(dh3)
        dp2 = dp2 * (p2 > 0)
        dh2 = maxpool_bwd(h2, dp2, 2) * (h2 > 0)
        dp1 = self.c2.bwd(dh2)
        dp1 = dp1 * (p1 > 0)
        dh1 = maxpool_bwd(h1, dp1, 2) * (h1 > 0)
        self.c1.bwd(dh1)
        return [self.c1.gw, self.c1.gb, self.c2.gw, self.c2.gb,
                self.c3.gw, self.c3.gb, gap.T @ flat, flat.sum(0)]

    def step(self, grads, lr, t, wd=1e-4):
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            g = g + wd * p
            m[:] = 0.9 * m + 0.1 * g
            v[:] = 0.999 * v + 0.001 * g * g
            p -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
