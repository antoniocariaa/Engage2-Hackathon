"""Caricamento dati: conversione CSV -> parquet (M0) e loader per train/validation/test."""
import argparse
import time

import numpy as np
import pandas as pd

from src.config import DATA_DIR, SAMPLE_ID_COL, ROW_INDEX_COL

SPLITS = ("train", "validation", "test")
STR_COLS = ["session", "timestamp", "task_name"]
INT_ID_COLS = ["sample_id", "seq_len", "row_index", "task_index", "atco", "scenario"]


def _downcast(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if c in STR_COLS:
            continue
        if c in INT_ID_COLS or c == "target_index":
            df[c] = df[c].astype(np.int32)
        elif pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype(np.float32)
    return df


def convert_all() -> None:
    for split in SPLITS:
        t0 = time.time()
        df = _downcast(pd.read_csv(DATA_DIR / f"{split}.csv"))
        df = df.sort_values([SAMPLE_ID_COL, ROW_INDEX_COL], kind="stable").reset_index(drop=True)
        out = DATA_DIR / f"{split}.parquet"
        df.to_parquet(out, index=False)
        print(f"{split:10s} {df.shape} -> {out.name} ({out.stat().st_size / 1e6:.1f} MB, {time.time() - t0:.1f}s)")


def load(split: str) -> pd.DataFrame:
    path = DATA_DIR / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} mancante: esegui `python -m src.data --convert`")
    return pd.read_parquet(path)


def load_trainval() -> pd.DataFrame:
    """train + validation concatenati, ordinati per sample_id contiguo (requisito dei group LTR)."""
    df = pd.concat([load("train"), load("validation")], ignore_index=True)
    return df.sort_values([SAMPLE_ID_COL, ROW_INDEX_COL], kind="stable").reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--convert", action="store_true", help="converte data/*.csv in data/*.parquet")
    args = ap.parse_args()
    if args.convert:
        convert_all()
    for split in SPLITS:
        t0 = time.time()
        df = load(split)
        print(f"load {split:10s} {df.shape} in {time.time() - t0:.2f}s")
