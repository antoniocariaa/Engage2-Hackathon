"""Tune causal smoothing (alpha, half_life) on CV OOF scores only; confirm once on validation.

  python -m src.tune_smoothing <experiment>
"""
from __future__ import annotations

import itertools
import sys
import time

import numpy as np
import pandas as pd

from src.cv import controller_folds, fold_array
from src.data_io import ROOT, load_all, load_config
from src.explog import log_experiment
from src.metrics import hits_from_scores, paired_session_bootstrap
from src.temporal import History, smooth

ALPHAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
HALF_LIVES = [3.0, 6.0, 12.0, 24.0, 48.0]


def main(name: str) -> None:
    cfg = load_config()
    d = load_all(cfg)
    tr, va = d["train"], d["validation"]
    pred = np.load(ROOT / "cache" / "preds" / f"{name}.npz")
    folds = fold_array(tr, controller_folds(tr, cfg["n_folds"]))
    htr, hva = History(tr), History(va)
    rows = []
    t0 = time.time()
    for a, hl in itertools.product(ALPHAS, HALF_LIVES):
        if a == 0.0 and hl != HALF_LIVES[0]:
            continue
        s = smooth(tr, pred["oof"], a, hl, htr)
        fs = [hits_from_scores(tr[folds == k], s[folds == k]).mean() for k in range(cfg["n_folds"])]
        rows.append(dict(alpha=a, half_life=hl, cv=np.mean(fs), **{f"f{k}": v for k, v in enumerate(fs)}))
    res = pd.DataFrame(rows).sort_values("cv", ascending=False)
    base = res[res.alpha == 0.0].iloc[0]
    best = res.iloc[0]
    print(res.head(10).to_string(index=False))
    print(f"base (no smoothing) CV {base.cv:.4f}; grid took {time.time() - t0:.0f}s")
    n_better = sum(best[f"f{k}"] > base[f"f{k}"] for k in range(cfg["n_folds"]))
    sv = smooth(va, pred["val"], best.alpha, best.half_life, hva)
    hb, hs = hits_from_scores(va, pred["val"]), hits_from_scores(va, sv)
    sess = va.groupby("sample_id").session.first()
    diff, lo, hi, p = paired_session_bootstrap(hs, hb, sess)
    print(f"best alpha={best.alpha} hl={best.half_life}: CV {best.cv:.4f} (+{best.cv - base.cv:.4f}, "
          f"{n_better}/4 folds better) | val {hs.mean():.4f} vs {hb.mean():.4f} diff {diff:+.4f} [{lo:+.4f},{hi:+.4f}]")
    res.to_csv(ROOT / "cache" / "preds" / f"{name}_smoothing_grid.csv", index=False)
    log_experiment("6", f"{name}+smooth", dict(base=name, alpha=best.alpha, half_life=best.half_life,
                                                 grid=dict(alphas=ALPHAS, half_lives=HALF_LIVES)),
                   [best[f"f{k}"] for k in range(cfg["n_folds"])], float(hs.mean()),
                   notes=f"base CV {base.cv:.4f}, folds better {n_better}/4, val diff {diff:+.4f} "
                         f"[{lo:+.4f},{hi:+.4f}]")
    np.savez(ROOT / "cache" / "preds" / f"{name}+smooth.npz", alpha=best.alpha, half_life=best.half_life,
             oof=smooth(tr, pred["oof"], best.alpha, best.half_life, htr), val=sv)


if __name__ == "__main__":
    main(sys.argv[1])
