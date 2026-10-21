"""Artifacts -> submission. No training code paths.

  python -m src.predict <experiment> [--out submissions/submission.csv]
  python -m src.predict <exp_a>,<exp_b> [--weights 0.5,0.5]   # within-sample z-score blend of final artifacts
  python -m src.predict --val-from-cv <experiment>   # validation submission from the train-only model
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from src import features
from src.data_io import ROOT, load_config, load_split
from src.models import load as load_model, predict as model_predict
from src.rank import topk, within_sample_z, write_submission
from src.validate_submission import validate


def score_artifact(name: str, df: pd.DataFrame) -> np.ndarray:
    """Average raw scores over the saved seed models of one experiment."""
    cfg = load_config()
    d = ROOT / cfg["paths"]["artifacts_dir"] / name
    meta = json.loads((d / "meta.json").read_text())
    dff, feats = features.build(df, tuple(meta["groups"]), tuple(meta["spec"].get("drop", ())))
    if meta["spec"].get("te"):
        import pickle
        enc = pickle.loads((d / "te_encoder.pkl").read_bytes())
        cols = features.te_apply(dff, enc)
        dff, feats = features.add_cols(dff, cols), feats + list(cols)
    if feats != meta["feats"]:
        raise ValueError("feature schema at inference differs from the frozen training schema")
    score = np.zeros(len(dff))
    for s in meta["seeds"]:
        b = load_model(str(d / f"model_seed{s}"), meta["spec"].get("lib", "lgb"))
        score += model_predict(b, dff, feats) / len(meta["seeds"])
    return score


def spot_check(df: pd.DataFrame, top: pd.DataFrame, n: int = 5, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    ti = df.set_index(["sample_id", "row_index"])["task_name"]
    for s in rng.choice(top.index.to_numpy(), size=n, replace=False):
        names = [ti.loc[(s, r)] for r in top.loc[s].tolist()]
        print(f"  sample {s}: {top.loc[s].tolist()} -> {names}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--out", default="submissions/submission.csv")
    ap.add_argument("--val-from-cv", action="store_true")
    ap.add_argument("--weights", default=None)
    a = ap.parse_args()
    if a.val_from_cv:
        va = load_split("validation")
        names = a.name.split(",")
        if len(names) == 1:
            sc = np.load(ROOT / "cache" / "preds" / f"{a.name}.npz")["val"]
        else:
            w = [float(x) for x in a.weights.split(",")] if a.weights else [1.0 / len(names)] * len(names)
            sc = sum(wi * within_sample_z(va, np.load(ROOT / "cache" / "preds" / f"{n}.npz")["val"])
                     for n, wi in zip(names, w))
        out = ROOT / "submissions" / "validation_submission.csv"
        write_submission(topk(va, sc), out)
        validate(out, "validation")
        return
    te = load_split("test")
    t0 = time.time()
    names = a.name.split(",")
    w = [float(x) for x in a.weights.split(",")] if a.weights else [1.0 / len(names)] * len(names)
    if len(names) == 1:
        sc = score_artifact(names[0], te)
    else:
        sc = sum(wi * within_sample_z(te, score_artifact(n, te)) for n, wi in zip(names, w))
    top = topk(te, sc)
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    write_submission(top, out)
    print(f"predicted {len(top)} samples with {names} weights {w} in {time.time() - t0:.1f}s -> {out}")
    validate(out, "test")  # re-read from disk
    spot_check(te, top)


if __name__ == "__main__":
    main()
