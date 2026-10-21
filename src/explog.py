"""Append-only experiment log -> reports/experiments.csv."""
from __future__ import annotations

import fcntl
import hashlib
import json
import time

import pandas as pd

from src.data_io import ROOT

LOG = ROOT / "reports" / "experiments.csv"


def config_hash(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:10]


def log_experiment(stage: str, name: str, cfg: dict, fold_scores: list[float] | None = None,
                   val_score: float | None = None, train_time: float | None = None,
                   infer_time: float | None = None, notes: str = "", **extra) -> dict:
    row = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), stage=stage, name=name, config_hash=config_hash(cfg),
               config=json.dumps(cfg, sort_keys=True, default=str))
    if fold_scores is not None:
        s = pd.Series(fold_scores, dtype=float)
        row.update(cv_mean=round(s.mean(), 5), cv_std=round(s.std(ddof=1), 5) if len(s) > 1 else None)
        for i, v in enumerate(fold_scores):
            row[f"fold{i}"] = round(float(v), 5)
    row.update(val=None if val_score is None else round(float(val_score), 5),
               train_time_s=None if train_time is None else round(train_time, 2),
               infer_time_s=None if infer_time is None else round(infer_time, 3), notes=notes)
    row.update({k: v for k, v in extra.items()})
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG.with_suffix(".lock"), "w") as lock:  # several experiment processes may log concurrently
        fcntl.flock(lock, fcntl.LOCK_EX)
        df = pd.DataFrame([row])
        if LOG.exists():
            old = pd.read_csv(LOG)
            df = pd.concat([old, df], ignore_index=True)
        df.to_csv(LOG, index=False)
    return row
