"""Paired comparison of two experiments under the §6.5 adoption rule -> reports/comparisons.csv

  python -m src.compare <candidate> <reference>
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from src.cv import controller_folds, fold_array
from src.data_io import ROOT, load_all, load_config
from src.metrics import hits_from_scores, paired_session_bootstrap

PRED = ROOT / "cache" / "preds"


def compare(cand: str, ref: str, data=None, cfg=None) -> dict:
    cfg = cfg or load_config()
    d = data or load_all(cfg)
    tr, va = d["train"], d["validation"]
    folds = fold_array(tr, controller_folds(tr, cfg["n_folds"]))
    a, b = np.load(PRED / f"{cand}.npz"), np.load(PRED / f"{ref}.npz")
    ha, hb = hits_from_scores(tr, a["oof"]), hits_from_scores(tr, b["oof"])
    fold_of = pd.Series(folds, index=tr["sample_id"]).groupby(level=0).first()
    fd = [float(ha[fold_of == k].mean() - hb[fold_of == k].mean()) for k in range(cfg["n_folds"])]
    s_tr = tr.groupby("sample_id").session.first()
    o_diff, o_lo, o_hi, _ = paired_session_bootstrap(ha, hb, s_tr)
    va_a, va_b = hits_from_scores(va, a["val"]), hits_from_scores(va, b["val"])
    v_diff, v_lo, v_hi, _ = paired_session_bootstrap(va_a, va_b, va.groupby("sample_id").session.first())
    fold_std = float(np.std([ha[fold_of == k].mean() for k in range(cfg["n_folds"])], ddof=1))
    n_better = int(sum(x > 0 for x in fd))
    adopt = (o_diff > 0) and n_better >= 3 and v_diff > -0.01
    row = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), candidate=cand, reference=ref,
               cv_diff=round(o_diff, 5), cv_diff_ci=f"[{o_lo:+.4f},{o_hi:+.4f}]",
               fold_diffs=str([round(x, 4) for x in fd]), folds_better=n_better, cand_fold_std=round(fold_std, 4),
               val_diff=round(v_diff, 5), val_diff_ci=f"[{v_lo:+.4f},{v_hi:+.4f}]", passes_6_5=adopt)
    path = ROOT / "reports" / "comparisons.csv"
    df = pd.DataFrame([row])
    if path.exists():
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    df.to_csv(path, index=False)
    print(f"{cand} vs {ref}: CV diff {o_diff:+.4f} {row['cv_diff_ci']} folds {row['fold_diffs']} ({n_better}/4) | "
          f"val diff {v_diff:+.4f} {row['val_diff_ci']} | passes §6.5: {adopt}")
    return row


if __name__ == "__main__":
    compare(sys.argv[1], sys.argv[2])
