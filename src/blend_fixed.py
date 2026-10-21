"""Evaluate a pre-specified (untuned) equal-weight within-sample z-score blend on CV OOF and validation.

  python -m src.blend_fixed <exp_a> <exp_b> [...]
"""
from __future__ import annotations

import sys

import numpy as np

from src.compare import compare
from src.cv import controller_folds, fold_array
from src.data_io import ROOT, load_all, load_config
from src.explog import log_experiment
from src.metrics import hits_from_scores
from src.rank import within_sample_z

PRED = ROOT / "cache" / "preds"


def main(names: list[str]) -> None:
    cfg = load_config()
    d = load_all(cfg)
    tr, va = d["train"], d["validation"]
    folds = fold_array(tr, controller_folds(tr, cfg["n_folds"]))
    ps = [np.load(PRED / f"{n}.npz") for n in names]
    oof = sum(within_sample_z(tr, p["oof"]) for p in ps) / len(ps)
    val = sum(within_sample_z(va, p["val"]) for p in ps) / len(ps)
    fs = [float(hits_from_scores(tr[folds == k], oof[folds == k]).mean()) for k in range(cfg["n_folds"])]
    vs = float(hits_from_scores(va, val).mean())
    name = "eq:" + "+".join(names)
    np.savez(PRED / f"{name.replace(':', '_')}.npz", oof=oof, val=val, R=0, R_full=0)
    log_experiment("7", name, dict(names=names, weights="equal", method="z"), fs, vs, notes="pre-specified weights")
    print(f"{name}: CV {np.mean(fs):.4f} ± {np.std(fs, ddof=1):.4f} {[round(x, 4) for x in fs]} | val {vs:.4f}")
    for n in names:
        compare(name.replace(":", "_"), n, d, cfg)


if __name__ == "__main__":
    main(sys.argv[1:])
