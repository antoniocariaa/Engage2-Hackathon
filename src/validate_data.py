"""Stage 0 data audit -> reports/data_audit.md.  Run: python -m src.validate_data"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_io import (ROOT, ID_COLS, TASK_TYPES, feature_columns, load_all, load_config,
                         load_sample_submission)

EXPECTED = {
    "train": dict(sessions=34, samples=11923, rows=432578, cols=96),
    "validation": dict(sessions=3, samples=1252, rows=42110, cols=96),
    "test": dict(sessions=5, samples=1863, rows=66409, cols=95),
}


def aircraft_of(df: pd.DataFrame) -> pd.Series:
    """Subject callsign: strip the type prefix from task_name, keep the first callsign."""
    prefix = df["task_index"].map(lambda t: TASK_TYPES[int(t)] + "_")
    names = df["task_name"].astype(str)
    rest = [n[len(p):] if n.startswith(p) else None for n, p in zip(names, prefix)]
    return pd.Series([r.split("_")[0] if r is not None else None for r in rest], index=df.index, dtype="string")


def seconds_of_day(ts: pd.Series) -> np.ndarray:
    parts = ts.astype(str).str.split(":", expand=True).astype(int)
    return (parts[0] * 3600 + parts[1] * 60 + parts[2]).to_numpy()


def main() -> None:
    cfg = load_config()
    data = load_all(cfg)
    out: list[str] = ["# Stage 0 data audit", ""]
    problems: list[str] = []

    def check(cond: bool, msg: str) -> None:
        out.append(f"- {'OK' if cond else '**FAIL**'}: {msg}")
        if not cond:
            problems.append(msg)

    # --- columns
    out += ["## Columns", ""]
    tr_cols, te_cols = list(data["train"].columns), list(data["test"].columns)
    check(list(data["validation"].columns) == tr_cols, "validation columns == train columns")
    check(set(te_cols) == set(tr_cols) - {"target_index"}, "test columns == train columns minus target_index")
    feats = feature_columns(tr_cols)
    check(len(feats) == 86, f"feature column count = {len(feats)} (expected 86)")
    check(all(c in tr_cols for c in ID_COLS), "all 10 identifier/label columns present")
    groups = {p: [c for c in feats if c.startswith(p)] for p in ("x_", "k_", "a_", "l_")}
    out.append("- prefix group sizes: " + ", ".join(f"{p}{len(v)}" for p, v in groups.items())
               + f", other {len(feats) - sum(map(len, groups.values()))}")
    out.append("")

    # --- counts
    out += ["## Counts vs documentation", "", "| split | sessions | samples | rows | cols | controllers | sessions list |",
            "|---|---:|---:|---:|---:|---|---|"]
    for s, df in data.items():
        n = dict(sessions=df.session.nunique(), samples=df.sample_id.nunique(), rows=len(df), cols=df.shape[1])
        atcos = sorted(df.atco.unique().tolist())
        out.append(f"| {s} | {n['sessions']} | {n['samples']} | {n['rows']} | {n['cols']} | {atcos} | "
                   f"{', '.join(sorted(df.session.unique()))} |")
    out.append("")
    for s, df in data.items():
        n = dict(sessions=df.session.nunique(), samples=df.sample_id.nunique(), rows=len(df), cols=df.shape[1])
        for k, v in EXPECTED[s].items():
            check(n[k] == v, f"{s} {k} = {n[k]} (documented {v})")
    out.append("")

    # --- per-sample invariants
    out += ["## Per-sample invariants", ""]
    for s, df in data.items():
        g = df.groupby("sample_id", sort=False)
        check((g.size() == g["seq_len"].first()).all(), f"{s}: rows per sample == seq_len")
        rmin, rmax, rnu = g["row_index"].min(), g["row_index"].max(), g["row_index"].nunique()
        seq = g["seq_len"].first()
        check(((rmin == 0) & (rmax == seq - 1) & (rnu == seq)).all(), f"{s}: row_index is exactly 0..seq_len-1")
        const = [c for c in ("seq_len", "session", "atco", "scenario", "timestamp") if (g[c].nunique() > 1).any()]
        check(not const, f"{s}: seq_len/session/atco/scenario/timestamp constant within sample (violations: {const})")
        check((g["task_name"].nunique() == seq).all(), f"{s}: task_name unique within sample")
        out.append(f"- {s}: seq_len range {seq.min()}–{seq.max()}, mean {seq.mean():.2f}")
        if "target_index" in df:
            pos = g["target_index"].sum()
            check((pos == 1).all(), f"{s}: exactly one target_index==1 per sample (counts: {pos.value_counts().to_dict()})")
            check(set(df.target_index.unique()) <= {0, 1}, f"{s}: target_index in {{0,1}}")
    out.append("")

    # --- task_index
    out += ["## Task types", ""]
    for s, df in data.items():
        vals = sorted(df.task_index.unique().tolist())
        out.append(f"- {s}: task_index values {vals}")
        prefix_ok = np.array([n.startswith(TASK_TYPES[int(t)] + "_")
                              for n, t in zip(df.task_name.astype(str), df.task_index)])
        check(prefix_ok.all(), f"{s}: task_name prefix matches task_index ({(~prefix_ok).sum()} mismatches)")
        check(11 not in vals, f"{s}: task type 11 absent")
    tr = data["train"]
    pos = tr[tr.target_index == 1].task_index.value_counts()
    rows = tr.task_index.value_counts()
    out += ["", "| task_index | type | positives (train) | rows (train) | pos rate |", "|---:|---|---:|---:|---:|"]
    for t in rows.sort_index().index:
        p = int(pos.get(t, 0))
        out.append(f"| {t} | {TASK_TYPES[int(t)]} | {p} | {rows[t]} | {p / rows[t]:.4f} |")
    out.append("")

    # --- aircraft-level constancy
    out += ["## Aircraft-level columns (within sample, same subject callsign)", ""]
    df = tr.copy()
    df["aircraft"] = aircraft_of(df)
    check(df.aircraft.notna().all(), "subject callsign parsed for every train row")
    multi = df[df.groupby(["sample_id", "aircraft"]).task_name.transform("size") > 1]
    nun = multi.groupby(["sample_id", "aircraft"])[feats].nunique(dropna=False)
    vary = (nun > 1).mean().sort_values(ascending=False)
    vary = vary[vary > 0]
    out.append(f"- groups (sample, aircraft) with >1 task: {len(nun)}")
    out.append("- columns that vary between tasks of the same aircraft (share of such groups):")
    for c, v in vary.items():
        out.append(f"  - `{c}`: {v:.3f}")
    twins = df[df.l_twin_AE == 1]
    twins = twins[twins.task_index.isin([2, 4])]
    tw_nun = twins.groupby(["sample_id", "aircraft"])[feats].nunique(dropna=False)
    tw_vary = (tw_nun > 1).mean()
    tw_vary = tw_vary[tw_vary > 0].sort_values(ascending=False)
    out.append(f"- assume/entry_coordination twins ({len(tw_nun)} pairs): varying columns "
               + ", ".join(f"`{c}` {v:.3f}" for c, v in tw_vary.items()))
    tpos = tr[tr.target_index == 1]
    tw_t = tpos[tpos.l_twin_AE == 1]
    out.append(f"- train targets with l_twin_AE=1: {len(tw_t)} / {len(tpos)}; by type "
               f"{tw_t.task_index.value_counts().to_dict()}")
    out.append("")

    # --- missingness
    out += ["## Missingness (fraction NaN)", "", "| column | train | validation | test | train is_assumed=1 | train is_assumed=0 |",
            "|---|---:|---:|---:|---:|---:|"]
    for c in feats:
        r = [data[s][c].isna().mean() for s in ("train", "validation", "test")]
        a1 = tr.loc[tr.is_assumed == 1, c].isna().mean()
        a0 = tr.loc[tr.is_assumed == 0, c].isna().mean()
        if max(r) > 0:
            out.append(f"| `{c}` | {r[0]:.3f} | {r[1]:.3f} | {r[2]:.3f} | {a1:.3f} | {a0:.3f} |")
    for s in data:
        check(data[s]["is_assumed"].notna().all(), f"{s}: is_assumed has no NaN")
    out.append("")

    # --- value ranges
    out += ["## Value checks", ""]
    for s, d in data.items():
        binary = [c for c in feats if set(pd.unique(d[c].dropna())) <= {0.0, 1.0}]
        age_cols = [c for c in feats if c.endswith("age") or c.endswith("AgeS") or c.startswith("a_since")]
        neg = {c: int((d[c] < 0).sum()) for c in age_cols if (d[c] < 0).any()}
        check(not neg, f"{s}: no negative ages/since columns (violations: {neg})")
        big = int((d.non_conformance_agreed_route_deviation_horizontally.abs() > 100_000).sum())
        out.append(f"- {s}: rows with |horizontal route deviation| > 100 km: {big}")
        out.append(f"- {s}: {len(binary)} columns are binary {{0,1}} (ignoring NaN)")
        dup = int(d.drop(columns=["sample_id"]).duplicated().sum())
        out.append(f"- {s}: fully duplicated rows (ignoring sample_id): {dup}")
        tts = d.time_to_sector
        out.append(f"- {s}: time_to_sector range [{tts.min():.0f}, {tts.max():.0f}]")
    out.append("")

    # --- temporal ordering
    out += ["## Timestamps and sessions", ""]
    for s, d in data.items():
        smp = d.groupby("sample_id", sort=True).first()[["session", "timestamp"]]
        smp["sec"] = seconds_of_day(smp.timestamp)
        diffs = smp.groupby("session")["sec"].diff().dropna()
        wraps = int((diffs < -43200).sum())
        nonmono = int(((diffs <= 0) & (diffs >= -43200)).sum())
        check(nonmono == 0, f"{s}: timestamps strictly increasing with sample_id within each session "
                            f"(non-increasing steps: {nonmono}, midnight wraps: {wraps})")
        gaps = diffs[diffs > 0]
        out.append(f"- {s}: gap between consecutive samples: =3s {np.mean(gaps == 3):.3f}, "
                   f"median {gaps.median():.0f}s, max {gaps.max():.0f}s")
        sess_contig = smp.groupby("session").apply(lambda x: np.all(np.diff(x.index.to_numpy()) == 1))
        out.append(f"- {s}: sample_ids contiguous per session: {bool(sess_contig.all())}")
    sess = {s: set(d.session.unique()) for s, d in data.items()}
    check(not (sess["train"] & sess["validation"] or sess["train"] & sess["test"] or sess["validation"] & sess["test"]),
          "sessions disjoint across files")
    at = {s: set(d.atco.unique()) for s, d in data.items()}
    out.append(f"- controller overlap train∩val {sorted(at['train'] & at['validation'])}, "
               f"train∩test {sorted(at['train'] & at['test'])}, val∩test {sorted(at['validation'] & at['test'])}")
    out.append("")

    # --- label persistence (descriptive only; never used as a feature)
    out += ["## Label persistence (descriptive)", ""]
    for s in ("train", "validation"):
        d = data[s]
        t = d[d.target_index == 1].sort_values("sample_id")[["sample_id", "session", "task_name"]].copy()
        same_sess = t.session.eq(t.session.shift()).fillna(False).astype(bool)
        same = (t.task_name.eq(t.task_name.shift()).fillna(False).astype(bool) & same_sess).sum() / same_sess.sum()
        prev_names = d.groupby("sample_id").task_name.apply(set)
        prev_t = t.task_name.shift()
        present = np.mean([pt in prev_names[sid] for sid, pt, ss in zip(t.sample_id, prev_t, same_sess) if ss])
        out.append(f"- {s}: consecutive same target task_name {same:.3f}; previous target still in list {present:.3f}")
        rel = (d.row_index / (d.seq_len - 1))[d.target_index == 1].mean()
        out.append(f"- {s}: mean relative target position row_index/(seq_len-1) = {rel:.3f}")
    out.append("")

    # --- sample submission
    out += ["## sample_submission.csv", ""]
    sub = load_sample_submission(cfg)
    tid = set(data["test"].sample_id.unique())
    check(list(sub.columns) == ["sample_id", "prediction"], "sample_submission header")
    check(set(sub.sample_id) == tid and not sub.sample_id.duplicated().any(),
          f"sample_submission ids == test ids ({len(sub)} rows, {len(tid)} test samples)")
    out.append(f"- first row: `{sub.iloc[0].sample_id},{sub.iloc[0].prediction}` (docs show `12797,9 6 4`)")
    out.append("")

    out.insert(2, f"**Failed checks: {len(problems)}**" + ("" if not problems else ": " + "; ".join(problems)))
    out.insert(3, "")
    path = ROOT / cfg["paths"]["reports_dir"] / "data_audit.md"
    path.write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
