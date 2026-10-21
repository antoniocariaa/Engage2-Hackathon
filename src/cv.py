"""Leave-controller-out folds on train.csv."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def controller_folds(df: pd.DataFrame, n_folds: int = 4) -> dict[int, int]:
    """atco -> fold id. GroupKFold on sample-level rows (balances samples per fold), deterministic."""
    smp = df.groupby("sample_id", sort=True)[["atco"]].first()
    gkf = GroupKFold(n_splits=n_folds)
    mapping: dict[int, int] = {}
    for f, (_, va) in enumerate(gkf.split(smp, groups=smp["atco"])):
        for a in smp["atco"].iloc[va].unique():
            mapping[int(a)] = f
    return mapping


def fold_array(df: pd.DataFrame, mapping: dict[int, int]) -> np.ndarray:
    return df["atco"].map(mapping).to_numpy().astype(int)


def fold_report(df: pd.DataFrame, mapping: dict[int, int]) -> pd.DataFrame:
    f = fold_array(df, mapping)
    rows = []
    for k in sorted(set(mapping.values())):
        for part, m in (("train", f != k), ("eval", f == k)):
            d = df[m]
            rows.append(dict(fold=k, part=part, samples=d.sample_id.nunique(), rows=len(d),
                             sessions=d.session.nunique(), controllers=sorted(d.atco.unique().astype(int).tolist()),
                             scenarios=sorted(d.scenario.unique().astype(int).tolist())))
    return pd.DataFrame(rows)
