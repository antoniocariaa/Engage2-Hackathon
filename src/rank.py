"""Within-sample ranking, top-k extraction, submission I/O and score ensembling."""
from __future__ import annotations

import numpy as np
import pandas as pd

K = 3


def topk(df: pd.DataFrame, score: np.ndarray, k: int = K) -> pd.DataFrame:
    """Per sample: the k row_index values with the highest score; ties -> lower row_index first.

    Returns a frame indexed by sample_id with columns p0..p{k-1} (int).
    """
    score = np.asarray(score, dtype=np.float64)
    if len(score) != len(df):
        raise ValueError("score length does not match data")
    if np.isnan(score).any():
        raise ValueError("NaN scores")
    sid = df["sample_id"].to_numpy()
    ri = df["row_index"].to_numpy()
    order = np.lexsort((ri, -score, sid))
    sid_o, ri_o = sid[order], ri[order]
    first = np.r_[True, sid_o[1:] != sid_o[:-1]]
    start = np.maximum.accumulate(np.where(first, np.arange(len(sid_o)), 0))
    pos = np.arange(len(sid_o)) - start
    keep = pos < k
    out = pd.DataFrame({"sample_id": sid_o[keep], "pos": pos[keep], "row_index": ri_o[keep]})
    wide = out.pivot(index="sample_id", columns="pos", values="row_index")
    if wide.isna().any().any():
        raise ValueError(f"some samples have fewer than {k} candidates")
    wide.columns = [f"p{i}" for i in range(k)]
    return wide.astype(int)


def within_sample_rank(df: pd.DataFrame, score: np.ndarray) -> np.ndarray:
    """Normalised descending rank within sample in [0, 1] (1 = best)."""
    s = pd.Series(np.asarray(score, dtype=np.float64), index=df.index)
    r = s.groupby(df["sample_id"].to_numpy()).rank(ascending=True, method="average")
    n = df["seq_len"].to_numpy()
    return ((r - 1) / np.maximum(n - 1, 1)).to_numpy()


def within_sample_z(df: pd.DataFrame, score: np.ndarray) -> np.ndarray:
    s = pd.Series(np.asarray(score, dtype=np.float64), index=df.index)
    g = s.groupby(df["sample_id"].to_numpy())
    return ((s - g.transform("mean")) / (g.transform("std") + 1e-9)).to_numpy()


def to_submission(top: pd.DataFrame) -> pd.DataFrame:
    pred = top.astype(str).agg(" ".join, axis=1)
    return pd.DataFrame({"sample_id": top.index.astype(int), "prediction": pred.to_numpy()})


def write_submission(top: pd.DataFrame, path) -> pd.DataFrame:
    sub = to_submission(top)
    sub.to_csv(path, index=False)
    return sub
