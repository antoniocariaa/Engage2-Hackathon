"""Strict submission validator (covers the gaps in scorer.py).

  python -m src.validate_submission submissions/submission.csv [--split test|validation]
"""
from __future__ import annotations

import argparse
import csv

import pandas as pd

from src.data_io import ROOT, load_split


def validate(path, split: str = "test") -> pd.DataFrame:
    with open(path, newline="") as f:
        header = next(csv.reader(f))
    if header != ["sample_id", "prediction"]:
        raise ValueError(f"header must be exactly sample_id,prediction (got {header})")
    sub = pd.read_csv(path, dtype={"sample_id": str, "prediction": str}, keep_default_na=False)
    df = load_split(split)
    seq = df.groupby("sample_id")["seq_len"].first()
    if not sub["sample_id"].str.fullmatch(r"\d+").all():
        raise ValueError("non-integer sample_id")
    sid = sub["sample_id"].astype(int)
    if sid.duplicated().any():
        raise ValueError(f"duplicated sample_id: {sid[sid.duplicated()].tolist()[:5]}")
    if len(sub) != len(seq):
        raise ValueError(f"row count {len(sub)} != unique {split} sample_ids {len(seq)}")
    missing, extra = set(seq.index) - set(sid), set(sid) - set(seq.index)
    if missing or extra:
        raise ValueError(f"sample_id mismatch: missing {sorted(missing)[:5]}, extra {sorted(extra)[:5]}")
    for s, p in zip(sid, sub["prediction"]):
        toks = p.split(" ")
        if len(toks) != 3 or not all(t.isdigit() for t in toks):
            raise ValueError(f"sample {s}: prediction must be three space-separated integers, got {p!r}")
        ids = [int(t) for t in toks]
        if len(set(ids)) != 3:
            raise ValueError(f"sample {s}: ids not distinct {ids}")
        if not all(0 <= i < seq[s] for i in ids):
            raise ValueError(f"sample {s}: id out of range [0, {seq[s] - 1}]: {ids}")
    print(f"OK: {path} — {len(sub)} rows, header ok, ids complete/unique, 3 distinct in-range ids per sample")
    return sub


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--split", default="test", choices=["test", "validation"])
    a = ap.parse_args()
    validate(ROOT / a.path if not a.path.startswith("/") else a.path, a.split)
