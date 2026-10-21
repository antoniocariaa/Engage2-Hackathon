"""Typed loading of the AWARE CSVs with Parquet caching and schema checks."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]

ID_COLS = ["sample_id", "seq_len", "row_index", "target_index", "task_index",
           "session", "atco", "scenario", "timestamp", "task_name"]
INT_ID = {"sample_id": "int32", "seq_len": "int16", "row_index": "int16", "target_index": "int8",
          "task_index": "int16", "atco": "int16", "scenario": "int8"}
STR_ID = {"session": "string", "timestamp": "string", "task_name": "string"}

TASK_TYPES = {
    0: "entry_conflict_resolution", 1: "exit_conflict_resolution", 2: "entry_coordination",
    3: "exit_coordination", 4: "assume_aircraft", 5: "transfer_aircraft", 6: "conflict_resolution",
    7: "non_conformance_resolution", 8: "qos_improvement", 9: "entry_conditions", 10: "exit_conditions",
    11: "aircraft_request", 12: "zone_conflict", 13: "return_to_route",
}


def load_config(path: str | os.PathLike | None = None) -> dict:
    path = Path(path) if path else ROOT / "config" / "default.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def _csv_path(cfg: dict, split: str) -> Path:
    return ROOT / cfg["paths"]["data_dir"] / cfg["paths"][split]


def feature_columns(columns) -> list[str]:
    """The 86 documented feature columns, in file order (selected by name, never position)."""
    return [c for c in columns if c not in ID_COLS]


def load_split(split: str, cfg: dict | None = None, use_cache: bool = True) -> pd.DataFrame:
    """Load train/validation/test with explicit dtypes, sorted by (sample_id, row_index)."""
    cfg = cfg or load_config()
    cache = ROOT / cfg["paths"]["cache_dir"] / f"{split}.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)
    path = _csv_path(cfg, split)
    header = pd.read_csv(path, nrows=0).columns.tolist()
    dtypes = {}
    for c in header:
        if c in INT_ID:
            dtypes[c] = INT_ID[c]
        elif c in STR_ID:
            dtypes[c] = STR_ID[c]
        else:
            dtypes[c] = "float32"
    df = pd.read_csv(path, dtype=dtypes)
    for c in ("sample_id", "row_index", "seq_len", "session"):
        if df[c].isna().any():
            raise ValueError(f"{split}: unexpected NaN in identifier column {c}")
    df = df.sort_values(["sample_id", "row_index"], kind="stable").reset_index(drop=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


def load_all(cfg: dict | None = None) -> dict[str, pd.DataFrame]:
    cfg = cfg or load_config()
    return {s: load_split(s, cfg) for s in ("train", "validation", "test")}


def load_sample_submission(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    return pd.read_csv(_csv_path(cfg, "sample_submission"), dtype={"prediction": str})


def check_groups(df: pd.DataFrame) -> None:
    """Fail loudly if rows are not contiguous per sample or do not match seq_len."""
    sid = df["sample_id"].to_numpy()
    if np.any(np.diff(sid) < 0):
        raise ValueError("data not sorted by sample_id")
    sizes = df.groupby("sample_id", sort=True).size()
    seq = df.groupby("sample_id", sort=True)["seq_len"].first()
    if not (sizes == seq).all():
        raise ValueError("group sizes do not match seq_len")
