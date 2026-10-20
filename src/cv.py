"""Harness di cross-validation (M2): GroupKFold per atco su train+validation, OOF salvati, log esperimenti.

    python -m src.cv --model lgbm_rank --features v4 --exp-id m3-v4 --note "aircraft"
"""
import argparse
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.config import EXPERIMENTS_DIR, GROUP_COL, N_FOLDS, OOF_DIR
from src.data import load_trainval
from src.features import build_features
from src.metric import summary
from src.episodes import sample_weights
from src.models.cat_rank import CatRank
from src.models.lgbm_rank import NUM_BOOST_ROUND, LGBMRank, LGBMXendcg

MODELS = {m.name: m for m in (LGBMRank, LGBMXendcg, CatRank)}


def make_folds(df: pd.DataFrame, n_folds: int = N_FOLDS) -> np.ndarray:
    """Fold per riga; GroupKFold deterministico sugli atco."""
    fold = np.full(len(df), -1, dtype=np.int8)
    for k, (_, va) in enumerate(GroupKFold(n_splits=n_folds).split(df, groups=df[GROUP_COL])):
        fold[va] = k
    return fold


def run_cv(model_name: str, version: str, params=None, rounds: int = NUM_BOOST_ROUND, alpha: float = 0.0,
           verbose=True):
    t0 = time.time()
    raw = load_trainval()
    df, feats = build_features(raw, version)
    t_feat = time.time() - t0
    df["fold"] = make_folds(df)
    w = sample_weights(df, alpha) if alpha > 0 else None
    oof = np.zeros(len(df))
    curves, imps = [], []
    for k in range(N_FOLDS):
        tr, va = df["fold"].to_numpy() != k, df["fold"].to_numpy() == k
        m = MODELS[model_name](params)
        m.fit(df.loc[tr, feats], df.loc[tr, "target_index"], df.loc[tr, "sample_id"],
              df.loc[va, feats], df.loc[va, "target_index"], df.loc[va, "sample_id"], num_boost_round=rounds,
              weight=None if w is None else w[tr])
        oof[va] = m.predict(df.loc[va, feats])
        curves.append(m.curve)
        imps.append(m.importance(feats))
        if verbose:
            part = df.loc[va].assign(score=oof[va])
            print(f"fold {k}: atco={sorted(int(a) for a in part[GROUP_COL].unique())} "
                  f"hit3={summary(part, 'score')['score']:.4f}", flush=True)
    df["score"] = oof
    res = summary(df, "score")
    curve = np.mean(curves, axis=0) if curves[0] is not None else np.full(1, np.nan)
    res.update(rounds=rounds, alpha=alpha, best_round=int(np.nanargmax(curve)) + 1 if curves[0] is not None else 0,
               best_curve=float(np.nanmax(curve)),
               n_feats=len(feats), t_feat=t_feat, t_total=time.time() - t0,
               importance=pd.concat(imps, axis=1).mean(axis=1).sort_values(ascending=False))
    return df, res


def log_experiment(exp_id, milestone, idea, version, model_name, res, cmd):
    row = (f"| {exp_id} | {datetime.now():%Y-%m-%d %H:%M} | {milestone} | {idea} | {version} ({res['n_feats']}) | "
           f"{model_name} | {res['score']:.4f} (ATCO {res['group_mean']:.4f} ± {res['group_std']:.4f}) | "
           f"{res['group_min']:.3f}–{res['group_max']:.3f} | alpha {res['alpha']}, {res['rounds']} round (curva media max "
           f"{res['best_curve']:.4f} @ {res['best_round']}), "
           f"{res['t_total'] / 60:.1f} min · `{cmd}` |\n")
    with open(EXPERIMENTS_DIR / "EXPERIMENTS.md", "a") as f:
        f.write(row)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lgbm_rank", choices=list(MODELS))
    ap.add_argument("--features", default="v0")
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--milestone", default="M3")
    ap.add_argument("--note", default="")
    ap.add_argument("--rounds", type=int, default=NUM_BOOST_ROUND)
    ap.add_argument("--alpha", type=float, default=0.0, help="pesi per episodio 1/len^alpha (0 = nessun peso)")
    args = ap.parse_args()
    df, res = run_cv(args.model, args.features, rounds=args.rounds, alpha=args.alpha)
    print(f"\nCV hit3 = {res['score']:.4f} | per ATCO {res['group_mean']:.4f} ± {res['group_std']:.4f} "
          f"[{res['group_min']:.3f}, {res['group_max']:.3f}] | {res['t_total'] / 60:.1f} min")
    print(f"curva media hit3 (diagnostica, ottimista): max {res['best_curve']:.4f} al round {res['best_round']}")
    print(res["per_group"].round(4).to_string())
    print("\ntop-15 importance (gain):\n" + res["importance"].head(15).round(0).to_string())
    df[["sample_id", "row_index", "atco", "task_index", "target_index", "fold", "score"]].to_parquet(
        OOF_DIR / f"{args.exp_id}.parquet", index=False)
    res["per_group"].to_csv(OOF_DIR / f"{args.exp_id}_per_atco.csv")
    with open(OOF_DIR / f"{args.exp_id}.json", "w") as f:
        json.dump({"model": args.model, "features": args.features, "rounds": args.rounds, "alpha": args.alpha,
                   "hit3": res["score"]}, f)
    log_experiment(args.exp_id, args.milestone, args.note, args.features, args.model, res,
                   f"python -m src.cv --model {args.model} --features {args.features} --rounds {args.rounds} --alpha {args.alpha} "
                   f"--exp-id {args.exp_id}")
