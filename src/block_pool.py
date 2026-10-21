"""Pool within-sample probabilities of each task over its block of consecutive snapshots (uses later snapshots).
Mixing weight beta tuned on CV OOF only; validation scored once.

  python -m src.block_pool <experiment>
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src.cv import controller_folds, fold_array
from src.data_io import ROOT, load_all, load_config
from src.explog import log_experiment
from src.features import blocks
from src.metrics import hits_from_scores
from src.temporal import within_softmax

PRED = ROOT / "cache" / "preds"
BETAS = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]


def pool(df: pd.DataFrame, score: np.ndarray, beta: float, temp: float = 1.0) -> np.ndarray:
    p = within_softmax(df, score, temp)
    blk = df["sample_id"].map(blocks(df)["block"]).to_numpy()
    key = pd.MultiIndex.from_arrays([blk, df["task_name"].astype(str).to_numpy()]).factorize()[0]
    pm = pd.Series(p).groupby(key).transform("mean").to_numpy()
    return (1 - beta) * p + beta * pm


def main(name: str) -> None:
    cfg = load_config()
    d = load_all(cfg)
    tr, va = d["train"], d["validation"]
    folds = fold_array(tr, controller_folds(tr, cfg["n_folds"]))
    pr = np.load(PRED / f"{name}.npz")
    rows = []
    for b in BETAS:
        s = pool(tr, pr["oof"], b)
        fs = [float(hits_from_scores(tr[folds == k], s[folds == k]).mean()) for k in range(cfg["n_folds"])]
        rows.append(dict(beta=b, cv=np.mean(fs), folds=[round(x, 4) for x in fs]))
    res = pd.DataFrame(rows)
    print(res.to_string(index=False))
    best = res.sort_values("cv", ascending=False).iloc[0]
    sv = pool(va, pr["val"], best.beta)
    v0, v1 = hits_from_scores(va, pr["val"]).mean(), hits_from_scores(va, sv).mean()
    print(f"{name}: best beta {best.beta} CV {best.cv:.4f} (beta=0: {res.cv.iloc[0]:.4f}) | val {v1:.4f} (unpooled {v0:.4f})")
    out = f"{name}+bpool"
    np.savez(PRED / f"{out}.npz", oof=pool(tr, pr["oof"], best.beta), val=sv, beta=best.beta, R=0, R_full=0)
    log_experiment("6", out, dict(base=name, beta=float(best.beta), grid=BETAS, uses_future=True), best.folds, float(v1),
                   notes=f"block pooling of within-sample softmax; unpooled CV {res.cv.iloc[0]:.4f} val {v0:.4f}")


if __name__ == "__main__":
    main(sys.argv[1])
