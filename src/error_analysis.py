"""§12 error analysis on CV OOF and validation predictions -> reports/error_analysis.md

  python -m src.error_analysis <model_experiment> [<baseline_experiment>=rule_baseline]
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src import baselines
from src.cv import controller_folds, fold_array
from src.data_io import ROOT, TASK_TYPES, load_all, load_config
from src.features import aircraft_key
from src.metrics import true_rank
from src.rank import topk

SHORT = {k: v.replace("_resolution", "").replace("_aircraft", "") for k, v in TASK_TYPES.items()}


def sample_table(df: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    """One row per sample: true type, rank of truth, top-1 predicted type, slices."""
    rk = true_rank(df, score)
    tgt = df[df.target_index == 1].set_index("sample_id")
    top = topk(df, score, 1)["p0"]
    ti = df.set_index(["sample_id", "row_index"])["task_index"]
    t = pd.DataFrame(index=tgt.index)
    t["rank"] = rk.reindex(t.index)
    t["hit1"], t["hit3"] = (t["rank"] <= 1).astype(int), (t["rank"] <= 3).astype(int)
    t["true_type"] = tgt["task_index"].map(SHORT)
    t["pred1_type"] = ti.reindex(pd.MultiIndex.from_arrays([top.index, top.to_numpy()])).map(SHORT).to_numpy()
    for c in ("session", "atco", "scenario", "seq_len", "l_twin_AE", "l_n_flight"):
        t[c] = tgt[c]
    t["seq_bucket"] = pd.cut(t["seq_len"], [0, 20, 30, 40, 50, 60]).astype(str)
    t["type_group"] = np.select([tgt["task_index"].isin([0, 1, 6]), tgt["task_index"].isin([2, 3, 4, 5])],
                                ["conflict(0,1,6)", "coord/handover(2,3,4,5)"], "other")
    # twin confusion: truth is assume (entry_coord) and the same aircraft's entry_coord (assume) ranks higher
    ac = aircraft_key(df)
    sc = pd.Series(np.asarray(score, float), index=df.index)
    d = pd.DataFrame({"sid": df["sample_id"].to_numpy(), "ac": ac, "ti": df["task_index"].to_numpy(),
                      "y": df["target_index"].to_numpy(), "s": sc.to_numpy()})
    tw = d[d.ti.isin([2, 4])]
    piv = tw.pivot_table(index="ac", columns="ti", values=["s", "y"], aggfunc="first")
    piv = piv.dropna()
    piv["sid"] = tw.groupby("ac")["sid"].first().reindex(piv.index)
    conf = np.where(piv[("y", 4)] == 1, piv[("s", 2)] > piv[("s", 4)],
                    np.where(piv[("y", 2)] == 1, piv[("s", 4)] > piv[("s", 2)], np.nan))
    cs = pd.Series(conf, index=piv["sid"].to_numpy()).dropna()
    t["twin_confused"] = cs.groupby(level=0).max().reindex(t.index)
    return t


def slice_table(t: pd.DataFrame, by: str) -> pd.DataFrame:
    g = t.groupby(by)
    return pd.DataFrame({"n": g.size(), "top1": g["hit1"].mean().round(3), "top3": g["hit3"].mean().round(3)})


def md(df: pd.DataFrame) -> str:
    return df.to_markdown() if hasattr(df, "to_markdown") else df.to_string()


def to_md(df: pd.DataFrame) -> str:
    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for i, r in df.iterrows():
        lines.append("| " + " | ".join([str(i)] + [str(v) for v in r.to_numpy()]) + " |")
    return "\n".join(lines)


def main(name: str, base: str = "rule_baseline") -> None:
    cfg = load_config()
    d = load_all(cfg)
    tr, va = d["train"], d["validation"]
    folds = fold_array(tr, controller_folds(tr, cfg["n_folds"]))
    pred = np.load(ROOT / "cache" / "preds" / f"{name}.npz")
    rule_oof = np.zeros(len(tr))
    for k in range(cfg["n_folds"]):
        m = folds == k
        rule_oof[m] = baselines.rule_score(tr[~m], tr[m])
    sets = {f"{name} / CV OOF": (tr, pred["oof"]), f"{name} / validation": (va, pred["val"]),
            f"{base} / CV OOF": (tr, rule_oof), f"{base} / validation": (va, baselines.rule_score(tr, va))}
    out = ["# Error analysis", "", f"Model: `{name}` (OOF at the CV-selected iteration; validation from the "
           f"train-only fit). Baseline: `{base}`.", ""]
    for title, (df, sc) in sets.items():
        t = sample_table(df, sc)
        out += [f"## {title}", "", f"- samples {len(t)}: top-1 {t.hit1.mean():.4f}, top-3 {t.hit3.mean():.4f}"]
        miss = t[t.hit3 == 0]
        out.append(f"- misses {len(miss)}: near misses (true rank 4–5) {int((miss['rank'] <= 5).sum())}, "
                   f"rank 6–10 {int(((miss['rank'] > 5) & (miss['rank'] <= 10)).sum())}, "
                   f"deep (>10) {int((miss['rank'] > 10).sum())}")
        out.append(f"- true-rank distribution (share): " + ", ".join(
            f"{r}: {v:.3f}" for r, v in t["rank"].clip(upper=11).value_counts(normalize=True).sort_index().items())
            + " (11 = 11+)")
        tw = t[t.l_twin_AE == 1]
        out.append(f"- twin_AE samples {len(tw)}: top-3 {tw.hit3.mean():.4f} vs others {t[t.l_twin_AE != 1].hit3.mean():.4f};"
                   f" twin confusion rate (truth assume/entry_coord, sibling ranked higher) "
                   f"{t.twin_confused.mean():.4f} over {int(t.twin_confused.notna().sum())} samples")
        mf = t[t.l_n_flight > 1]
        out.append(f"- target aircraft has >1 task (l_n_flight>1): n={len(mf)}, top-3 {mf.hit3.mean():.4f}; "
                   f"single-task aircraft top-3 {t[t.l_n_flight == 1].hit3.mean():.4f}")
        out.append("")
        for by in ("true_type", "type_group", "seq_bucket", "scenario", "atco", "session"):
            out += [f"**by {by}**", "", to_md(slice_table(t, by)), ""]
        conf = pd.crosstab(t["true_type"], t["pred1_type"])
        out += ["**predicted top-1 type (columns) vs true type (rows)**", "", to_md(conf), ""]
    gain_name = name if (ROOT / "cache" / "preds" / f"{name}_gain.csv").exists() else "lgb_binary_3s"
    gain = pd.read_csv(ROOT / "cache" / "preds" / f"{gain_name}_gain.csv", index_col=0).iloc[:, 0]
    out += [f"## Feature importance (gain share, `{gain_name}` full-train fit)", "",
            to_md((gain / gain.sum()).round(4).head(30).to_frame("gain_share")), ""]
    path = ROOT / "reports" / "error_analysis.md"
    path.write_text("\n".join(out) + "\n")
    print("\n".join(out[:60]))


if __name__ == "__main__":
    main(*sys.argv[1:])
