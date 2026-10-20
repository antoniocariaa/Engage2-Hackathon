"""Episodi di lavoro (M5): snapshot consecutivi della stessa sessione con lo stesso task target.

Servono SOLO a pesare il training (usano le label): mai come feature, mai sul test.
"""
import numpy as np
import pandas as pd

MAX_GAP_S = 3.5


def episode_ids(df: pd.DataFrame) -> pd.Series:
    """episode id per sample_id: nuovo episodio se cambia il task target, la sessione, o il gap supera MAX_GAP_S."""
    tgt = df.loc[df["target_index"] == 1, ["sample_id", "session", "timestamp", "task_name"]]
    tgt = tgt.sort_values(["session", "sample_id"])
    t = pd.to_timedelta(tgt["timestamp"]).dt.total_seconds()
    new = ((tgt["session"] != tgt["session"].shift()) | (tgt["task_name"] != tgt["task_name"].shift())
           | (t.diff() > MAX_GAP_S))
    return pd.Series(new.cumsum().to_numpy(), index=tgt["sample_id"].to_numpy(), name="episode")


def sample_weights(df: pd.DataFrame, alpha: float = 0.5) -> np.ndarray:
    """Peso per riga = 1 / len(episodio)^alpha, normalizzato a media 1 sui sample; costante dentro lo snapshot."""
    ep = episode_ids(df)
    w = 1.0 / ep.map(ep.value_counts()).astype(float) ** alpha
    w = w / w.mean()
    return df["sample_id"].map(w).to_numpy(dtype=np.float32)


if __name__ == "__main__":
    from src.data import load_trainval
    ep = episode_ids(load_trainval())
    sizes = ep.value_counts()
    print(f"episodi: {len(sizes)}  sample: {len(ep)}  lunghezza (snapshot): {sizes.describe().round(1).to_dict()}")
