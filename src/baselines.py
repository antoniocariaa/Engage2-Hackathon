"""Non-learned baselines. Each returns a per-row score (higher = better) for ranking within sample."""
from __future__ import annotations

import numpy as np
import pandas as pd


def type_rate(train: pd.DataFrame) -> pd.Series:
    """Positive rate per task_index in the given (training) data."""
    return train.groupby("task_index")["target_index"].mean()


def random_score(df: pd.DataFrame, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).random(len(df))


def type_prior_score(train: pd.DataFrame, df: pd.DataFrame) -> np.ndarray:
    """score = positive rate of the row's type in the training data (ties -> row_index)."""
    return df["task_index"].map(type_rate(train)).fillna(0.0).to_numpy(dtype=np.float64)


def rule_score(train: pd.DataFrame, df: pd.DataFrame) -> np.ndarray:
    """Reproduces sample_submission.csv exactly (1863/1863 samples on test):
    types ordered by training positive rate (desc), then ascending |time_to_sector| within type,
    then ascending row_index."""
    rate = type_rate(train).sort_values(ascending=False, kind="stable")
    prio = pd.Series(np.arange(len(rate)), index=rate.index)
    p = df["task_index"].map(prio).fillna(len(rate)).to_numpy(dtype=np.float64)
    t = np.abs(df["time_to_sector"].to_numpy(dtype=np.float64))
    t = np.nan_to_num(t, nan=1e9) / 1e5  # |time_to_sector| <= 3600 -> < 0.04, so type order dominates
    return -(p + t)
