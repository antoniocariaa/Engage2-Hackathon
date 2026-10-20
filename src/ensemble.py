"""Ensemble sugli OOF (M6): normalizzazione per snapshot, pesi scelti con CV annidata sui fold per ATCO.

    python -m src.ensemble --oof abl-ac m4-xendcg-ac m4-cat-ac --out ens-1
"""
import argparse
import itertools
import json

import numpy as np
import pandas as pd

from src.config import EXPERIMENTS_DIR, OOF_DIR
from src.metric import hits, summary

STEP = 0.1


def normalize(df: pd.DataFrame, col: str) -> pd.Series:
    """z-score dentro lo snapshot: rende confrontabili score di modelli con scale diverse."""
    g = df.groupby("sample_id", sort=False)[col]
    return (df[col] - g.transform("mean")) / (g.transform("std") + 1e-9)


def load_oofs(names: list) -> pd.DataFrame:
    base = None
    for n in names:
        o = pd.read_parquet(OOF_DIR / f"{n}.parquet")
        o[n] = normalize(o, "score")
        o = o.drop(columns="score")
        base = o if base is None else base.merge(o[["sample_id", "row_index", n]], on=["sample_id", "row_index"])
    return base.sort_values(["sample_id", "row_index"]).reset_index(drop=True)


def weight_grid(k: int):
    steps = int(round(1 / STEP))
    for c in itertools.product(range(steps + 1), repeat=k):
        if sum(c) == steps:
            yield np.array(c) / steps


def blend(df, names, w):
    return df[names].to_numpy() @ w


def best_weights(df, names):
    best, best_w = -1, None
    for w in weight_grid(len(names)):
        s = hits(df.assign(_s=blend(df, names, w)), "_s").mean()
        if s > best:
            best, best_w = s, w
    return best_w, best


def nested_cv(df, names):
    """per ogni fold: pesi scelti sugli altri fold, applicati al fold tenuto fuori."""
    out = np.zeros(len(df))
    chosen = []
    for k in sorted(df["fold"].unique()):
        va = df["fold"].to_numpy() == k
        w, _ = best_weights(df[~va], names)
        out[va] = blend(df[va], names, w)
        chosen.append(w.tolist())
    return out, chosen


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    df = load_oofs(args.oof)
    rows = []
    for n in args.oof:
        r = summary(df, n)
        rows.append((n, r["score"], r["group_mean"], r["group_std"]))
    eq = np.ones(len(args.oof)) / len(args.oof)
    df["equal"] = blend(df, args.oof, eq)
    df["nested"], chosen = nested_cv(df, args.oof)
    w_all, _ = best_weights(df, args.oof)
    for n in ["equal", "nested"]:
        r = summary(df, n)
        rows.append((n, r["score"], r["group_mean"], r["group_std"]))
    print(pd.DataFrame(rows, columns=["modello", "hit3", "ATCO mean", "ATCO std"]).round(4).to_string(index=False))
    print("pesi per fold (nested):", chosen)
    print("pesi finali (tutti i fold):", dict(zip(args.oof, w_all.round(2).tolist())))
    df["final"] = blend(df, args.oof, w_all)
    df[["sample_id", "row_index", "atco", "task_index", "target_index", "fold"]].assign(score=df["nested"]).to_parquet(
        OOF_DIR / f"{args.out}.parquet", index=False)
    with open(EXPERIMENTS_DIR / f"ensemble_{args.out}.json", "w") as f:
        json.dump({"models": args.oof, "weights": w_all.tolist(), "nested_fold_weights": chosen,
                   "nested_hit3": summary(df, "nested")["score"], "equal_hit3": summary(df, "equal")["score"]}, f,
                  indent=2)
