"""Top-3 hit rate (reproduces scorer.py) and session-block bootstrap."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.rank import K, topk


def labels_of(df: pd.DataFrame) -> pd.Series:
    """sample_id -> labelled row_index."""
    lab = df.loc[df["target_index"] == 1, ["sample_id", "row_index"]]
    return lab.set_index("sample_id")["row_index"]


def parse_prediction(pred: str) -> list[int]:
    """Same parsing as scorer._ids: int(float(tok)), duplicates dropped, order kept."""
    out: list[int] = []
    for tok in str(pred).split():
        i = int(float(tok))
        if i not in out:
            out.append(i)
    return out


def hits_from_submission(sub: pd.DataFrame, labels: pd.Series) -> pd.Series:
    """Per-sample hit (0/1) for a submission frame (sample_id, prediction) against labels."""
    s = sub.set_index("sample_id")["prediction"]
    missing = set(labels.index) - set(s.index)
    if missing:
        raise ValueError(f"{len(missing)} labelled samples missing from submission")
    preds = s.reindex(labels.index)
    hits = [int(int(t) in parse_prediction(p)[:K]) for p, t in zip(preds, labels.to_numpy())]
    return pd.Series(hits, index=labels.index, dtype=np.int8)


def hits_from_top(top: pd.DataFrame, labels: pd.Series) -> pd.Series:
    """Vectorised per-sample hits from a topk() frame."""
    t = top.reindex(labels.index)
    if t.isna().any().any():
        raise ValueError("labelled samples missing from predictions")
    h = (t.to_numpy() == labels.to_numpy()[:, None]).any(axis=1)
    return pd.Series(h.astype(np.int8), index=labels.index)


def hits_from_scores(df: pd.DataFrame, score: np.ndarray, k: int = K) -> pd.Series:
    return hits_from_top(topk(df, score, k), labels_of(df))


def hit_rate(df: pd.DataFrame, score: np.ndarray, k: int = K) -> float:
    return float(hits_from_scores(df, score, k).mean())


def true_rank(df: pd.DataFrame, score: np.ndarray) -> pd.Series:
    """1-based rank of the labelled row within its sample (same tie-break as topk)."""
    score = np.asarray(score, dtype=np.float64)
    sid = df["sample_id"].to_numpy()
    ri = df["row_index"].to_numpy()
    order = np.lexsort((ri, -score, sid))
    sid_o = sid[order]
    first = np.r_[True, sid_o[1:] != sid_o[:-1]]
    start = np.maximum.accumulate(np.where(first, np.arange(len(sid_o)), 0))
    pos = np.empty(len(sid), dtype=np.int64)
    pos[order] = np.arange(len(sid_o)) - start + 1
    m = df["target_index"].to_numpy() == 1
    return pd.Series(pos[m], index=sid[m])


def session_bootstrap(hits: pd.Series, session_of: pd.Series, n_boot: int = 2000, seed: int = 0,
                      alpha: float = 0.05) -> tuple[float, float, float]:
    """Mean and (1-alpha) CI by resampling whole sessions (blocks) with replacement."""
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"h": hits.to_numpy(), "s": session_of.reindex(hits.index).to_numpy()})
    agg = d.groupby("s")["h"].agg(["sum", "count"])
    sums, cnts = agg["sum"].to_numpy(), agg["count"].to_numpy()
    idx = rng.integers(0, len(agg), size=(n_boot, len(agg)))
    stats = sums[idx].sum(1) / cnts[idx].sum(1)
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(hits.mean()), float(lo), float(hi)


def paired_session_bootstrap(h_a: pd.Series, h_b: pd.Series, session_of: pd.Series, n_boot: int = 2000,
                             seed: int = 0) -> tuple[float, float, float, float]:
    """Difference a - b with session-block bootstrap CI and P(diff <= 0)."""
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"a": h_a.to_numpy(), "b": h_b.reindex(h_a.index).to_numpy(),
                      "s": session_of.reindex(h_a.index).to_numpy()})
    agg = d.groupby("s").agg(a=("a", "sum"), b=("b", "sum"), n=("a", "count"))
    idx = rng.integers(0, len(agg), size=(n_boot, len(agg)))
    diff = (agg["a"].to_numpy()[idx].sum(1) - agg["b"].to_numpy()[idx].sum(1)) / agg["n"].to_numpy()[idx].sum(1)
    lo, hi = np.quantile(diff, [0.025, 0.975])
    return float(h_a.mean() - h_b.reindex(h_a.index).mean()), float(lo), float(hi), float((diff <= 0).mean())
