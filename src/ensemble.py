"""Stage 7: blend experiments by within-sample z-score (or rank) averaging. Weights from CV OOF only.

  python -m src.ensemble <exp_a> <exp_b> [<exp_c> ...] [--method z|rank]
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np

from src.cv import controller_folds, fold_array
from src.data_io import ROOT, load_all, load_config
from src.explog import log_experiment
from src.metrics import hits_from_scores, paired_session_bootstrap
from src.rank import within_sample_rank, within_sample_z

PRED = ROOT / "cache" / "preds"


def normalise(df, score, method):
    return within_sample_z(df, score) if method == "z" else within_sample_rank(df, score)


def blend(parts: list[np.ndarray], w: tuple[float, ...]) -> np.ndarray:
    return sum(wi * p for wi, p in zip(w, parts))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--method", default="z", choices=["z", "rank"])
    a = ap.parse_args()
    cfg = load_config()
    d = load_all(cfg)
    tr, va = d["train"], d["validation"]
    folds = fold_array(tr, controller_folds(tr, cfg["n_folds"]))
    preds = [np.load(PRED / f"{n}.npz") for n in a.names]
    oof = [normalise(tr, p["oof"], a.method) for p in preds]
    val = [normalise(va, p["val"], a.method) for p in preds]

    def fold_scores(s):
        return [float(hits_from_scores(tr[folds == k], s[folds == k]).mean()) for k in range(cfg["n_folds"])]

    singles = {n: fold_scores(o) for n, o in zip(a.names, oof)}
    best_single = max(singles, key=lambda n: np.mean(singles[n]))
    grid = [w for w in itertools.product(np.arange(0, 1.01, 0.1), repeat=len(a.names)) if abs(sum(w) - 1) < 1e-9]
    res = []
    for w in grid:
        fs = fold_scores(blend(oof, w))
        res.append((np.mean(fs), w, fs))
    res.sort(key=lambda r: -r[0])
    cv_mean, w, fs = res[0]
    ref = singles[best_single]
    n_better = sum(x > y for x, y in zip(fs, ref))
    sess = va.groupby("sample_id").session.first()
    hb = hits_from_scores(va, blend(val, w))
    hs = hits_from_scores(va, val[a.names.index(best_single)])
    diff, lo, hi, p = paired_session_bootstrap(hb, hs, sess)
    for n, s in singles.items():
        print(f"  single {n}: CV {np.mean(s):.4f} {[round(x, 4) for x in s]}")
    print(f"blend {a.method} w={tuple(round(x, 2) for x in w)}: CV {cv_mean:.4f} {[round(x, 4) for x in fs]} "
          f"(+{cv_mean - np.mean(ref):.4f} vs {best_single}, {n_better}/4 folds better) | val {hb.mean():.4f} vs "
          f"{hs.mean():.4f} diff {diff:+.4f} [{lo:+.4f},{hi:+.4f}]")
    name = "blend:" + "+".join(a.names) + f":{a.method}"
    log_experiment("7", name, dict(names=a.names, method=a.method, weights=list(map(float, w))), fs,
                   float(hb.mean()), notes=f"vs {best_single} CV {np.mean(ref):.4f}; folds better {n_better}/4; "
                                         f"val diff {diff:+.4f} [{lo:+.4f},{hi:+.4f}]")
    np.savez(PRED / f"{name.replace(':', '_')}.npz", oof=blend(oof, w), val=blend(val, w), weights=np.array(w),
             names=np.array(a.names), method=a.method)


if __name__ == "__main__":
    main()
