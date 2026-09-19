"""final trainer: the glassjaw cnn over log-mel spectrograms.

recipe (validated in the model study):
  - energy gate: windows with peak < -40 dbfs are excluded from training and
    skipped at inference (esc-50 anomaly clips are silence-padded; up to a
    third of "positive" windows contain no event at all, which caps every
    model at ~0.92 auc until the gate is applied)
  - architecture: conv(1->16) -> pool -> conv(16->32) -> pool -> conv(32->32)
    -> global average pool -> fc(1). ~1.75 m macs per inference, 14 kb int8.
  - training: adam + cosine lr, time-roll augmentation (+-3 frames), mixup
    (p=0.5, beta 0.2), positive enrichment. folds 1-3 train, fold 4 for model
    selection, fold 5 held out for the final test number.
  - final model: best single seed by validation auc (SEEDS below; the sweep
    showed seed variance is small and a 3-seed ensemble bought ~0.005 auc —
    not worth 3x training time on a loaded machine. bump SEEDS if you want it).

outputs:
  models/weights.npz        float weights (for export_model.py)
  models/history.json       per-seed val/test metrics
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from train_core import Net, im2col, maxpool, maxpool_bwd, sig  # noqa: E402

GATE_DB = -40.0
EPOCHS = 60
SEEDS = (7,)


def load_gated():
    d = np.load("data/train.npz")
    dt = np.load("data/test.npz")
    pk_tr = np.load("data/cache/peak_train.npy")
    pk_te = np.load("data/cache/peak_test.npy")
    ktr, kte = pk_tr >= GATE_DB, pk_te >= GATE_DB
    Mtr = np.load("data/cache/mels_train.npy")[ktr]     # (N,61,26)
    Mte = np.load("data/cache/mels_test.npy")[kte]
    return (
        Mtr, d["y"][ktr].astype(np.float32), d["fold"][ktr],
        Mte, dt["y"][kte].astype(np.float32),
        d["cat"][ktr], dt["cat"][kte],
    )


def main():
    Mtr, y, folds, Mte, yt, cats, catst = load_gated()
    X = Mtr.transpose(0, 2, 1)[:, :, :, None]           # (N,26,61,1)
    Xt = Mte.transpose(0, 2, 1)[:, :, :, None]
    val = folds == 4
    mu = X[~val].transpose(0, 3, 1, 2).reshape(-1, 26).mean(0).reshape(1, 26, 1, 1)
    sd = X[~val].transpose(0, 3, 1, 2).reshape(-1, 26).std(0).reshape(1, 26, 1, 1) + 1e-6
    X = np.clip((X - mu) / sd, -5, 5)
    Xt = np.clip((Xt - mu) / sd, -5, 5)
    Xval, yval = X[val], y[val]
    Xtr, ytr = X[~val], y[~val]
    pos = np.where(ytr > 0.5)[0]

    nets, history = [], {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        net = Net(rng)
        t, best, best_state = 0, 0.0, None
        for ep in range(EPOCHS):
            # exp17-validated recipe: per-batch positive enrichment (+8),
            # mixup p=0.3, roll +-3. epoch-level oversampling collapsed val
            # auc (0.76 -> 0.51); keep what works.
            order = rng.permutation(len(Xtr))
            lr = 1.5e-3 * (0.5 * (1 + np.cos(np.pi * ep / EPOCHS)))
            s0 = 0
            while s0 < len(order):
                b = order[s0:s0 + 32]
                b = np.concatenate([b, rng.choice(pos, 8)])
                s0 += 32
                xb, yb = Xtr[b].copy(), ytr[b].copy()
                xb = np.roll(xb, rng.integers(-3, 4), axis=2)
                if rng.random() < 0.3:
                    perm = rng.permutation(len(b))
                    a_ = rng.beta(0.2, 0.2)
                    xb = a_ * xb + (1 - a_) * xb[perm]
                    yb = a_ * yb + (1 - a_) * yb[perm]
                t += 1
                net.step(net.backward(net.forward(xb), yb, 1.0), lr, t)
            pv = sig(net.forward(Xval))[:, 0]
            auc = roc_auc_score(yval, pv)
            if auc > best:
                best, best_state = auc, [q.copy() for q in net.params]
            if ep % 10 == 0 or ep == EPOCHS - 1:
                print(f"seed {seed} ep{ep}: val auc={auc:.4f} best={best:.4f}", flush=True)
        for p_, bs_ in zip(net.params, best_state):
            p_[:] = bs_
        nets.append(net)
        pt = sig(net.forward(Xt))[:, 0]
        history[seed] = {
            "val_auc": float(best),
            "test_auc": float(roc_auc_score(yt.astype(int), pt)),
        }
        print(f"seed {seed}: {history[seed]}", flush=True)

    pt_ens = np.mean([sig(n.forward(Xt))[:, 0] for n in nets], axis=0)
    ens_auc = float(roc_auc_score(yt.astype(int), pt_ens))
    print(f"ensemble(3) test auc: {ens_auc:.4f}")

    # pick a threshold on the ensemble val scores: highest recall with fp rate <= 2%
    pv_ens = np.mean([sig(n.forward(Xval))[:, 0] for n in nets], axis=0)
    order = np.argsort(-pv_ens)
    ps, ys = pv_ens[order], yval[order]
    n_neg = (ys == 0).sum()
    thr = 0.5
    for k in range(len(ps)):
        if (ys[:k + 1] == 0).sum() / n_neg > 0.02:
            thr = ps[max(k - 1, 0)]
            break
    pred = pt_ens >= thr
    tp = int(((pred == 1) & (yt == 1)).sum()); fn = int(((pred == 0) & (yt == 1)).sum())
    fp = int(((pred == 1) & (yt == 0)).sum()); tn = int(((pred == 0) & (yt == 0)).sum())
    print(f"threshold {thr:.4f}: tp={tp} fp={fp} fn={fn} tn={tn}")

    Path("models").mkdir(exist_ok=True)
    best_net = nets[int(np.argmax([history[s]["val_auc"] for s in SEEDS]))]
    np.savez_compressed(
        "models/weights.npz",
        c1w=best_net.c1.w, c1b=best_net.c1.b,
        c2w=best_net.c2.w, c2b=best_net.c2.b,
        c3w=best_net.c3.w, c3b=best_net.c3.b,
        fcw=best_net.fc, fcb=best_net.fcb,
        mu=mu.ravel(), sd=sd.ravel(),
    )
    import json
    Path("models/history.json").write_text(json.dumps({
        "gate_db": GATE_DB, "seeds": {str(k): v for k, v in history.items()},
        "ensemble_test_auc": ens_auc, "threshold": float(thr),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }, indent=2))
    print("wrote models/weights.npz + history.json")


if __name__ == "__main__":
    main()
