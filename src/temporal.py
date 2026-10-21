"""Causal score smoothing across snapshots of the same session (strictly earlier snapshots only).

s'_t = p_t + alpha * sum_{k: earlier snapshots containing the same task_name} 0.5**((T_t - T_k)/half_life) * p_k
where p is the within-sample softmax of the model score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import seconds


def within_softmax(df: pd.DataFrame, score: np.ndarray, temp: float = 1.0) -> np.ndarray:
    s = pd.Series(np.asarray(score, dtype=np.float64) / temp, index=df.index)
    g = s.groupby(df["sample_id"].to_numpy())
    e = np.exp(s - g.transform("max"))
    return (e / e.groupby(df["sample_id"].to_numpy()).transform("sum")).to_numpy()


class History:
    """Precomputed ordering of rows by (session, task_name, time) for fast repeated smoothing."""

    def __init__(self, df: pd.DataFrame):
        sec = pd.Series(seconds(df["timestamp"]), index=df.index)
        key = pd.MultiIndex.from_arrays([df["session"].astype(str).to_numpy(),
                                         df["task_name"].astype(str).to_numpy()]).factorize()[0]
        self.order = np.lexsort((sec.to_numpy(), key))
        self.key = key[self.order]
        self.t = sec.to_numpy()[self.order].astype(np.float64)
        self.same = np.r_[False, self.key[1:] == self.key[:-1]]
        self.dt = np.clip(np.r_[0.0, np.diff(self.t)], 0.0, None)  # negative only across keys (masked)

    def decayed_sum(self, p: np.ndarray, half_life: float) -> np.ndarray:
        """h_t = sum over earlier occurrences k of the same task of 0.5**((T_t-T_k)/hl) * p_k."""
        po = p[self.order]
        f = np.where(self.same, 0.5 ** (self.dt / half_life), 0.0)
        h = np.zeros(len(po))
        same, ff = self.same, f
        acc = 0.0
        for i in range(len(po)):
            acc = ff[i] * (acc + po[i - 1]) if same[i] else 0.0
            h[i] = acc
        out = np.empty_like(h)
        out[self.order] = h
        return out


def smooth(df: pd.DataFrame, score: np.ndarray, alpha: float, half_life: float, hist: History | None = None,
           temp: float = 1.0) -> np.ndarray:
    hist = hist or History(df)
    p = within_softmax(df, score, temp)
    return p + alpha * hist.decayed_sum(p, half_life)
