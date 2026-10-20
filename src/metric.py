"""Top-3 hit rate, equivalente a scorer.py ma calcolato direttamente su un DataFrame a righe."""
import numpy as np
import pandas as pd

K = 3


def top_k(df: pd.DataFrame, score_col: str, k: int = K) -> pd.DataFrame:
    """Prime k righe per sample_id, score decrescente; pareggi risolti per row_index crescente (deterministico)."""
    return (df.sort_values(["sample_id", score_col, "row_index"], ascending=[True, False, True], kind="stable")
              .groupby("sample_id", sort=False).head(k))


def hits(df: pd.DataFrame, score_col: str, k: int = K) -> pd.Series:
    """1/0 per sample_id: il target è tra le prime k righe."""
    return top_k(df, score_col, k).groupby("sample_id", sort=False)["target_index"].max()


def top3_hit_rate(df: pd.DataFrame, score_col: str) -> float:
    return float(hits(df, score_col).mean())


def top3_by_group(df: pd.DataFrame, score_col: str, group_col: str = "atco") -> pd.Series:
    h = hits(df, score_col)
    groups = df.drop_duplicates("sample_id").set_index("sample_id")[group_col]
    return h.groupby(groups.loc[h.index].values).mean()


def to_submission(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """sample_id,prediction con i 3 row_index migliori separati da spazio."""
    top = top_k(df, score_col)
    pred = top.groupby("sample_id", sort=False)["row_index"].agg(lambda s: " ".join(map(str, s)))
    return pred.rename("prediction").reset_index()


def summary(df: pd.DataFrame, score_col: str, group_col: str = "atco") -> dict:
    per = top3_by_group(df, score_col, group_col)
    return {"score": top3_hit_rate(df, score_col), "group_mean": float(per.mean()), "group_std": float(per.std()),
            "group_min": float(per.min()), "group_max": float(per.max()), "per_group": per}


def random_expectation(df: pd.DataFrame) -> float:
    """Hit rate atteso con un ranking casuale: media di min(3, n)/n."""
    n = df.groupby("sample_id").size()
    return float(np.mean(np.minimum(K, n) / n))
