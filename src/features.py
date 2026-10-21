"""Feature construction shared by train / validation / test. Labels are never read here."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_io import ID_COLS, TASK_TYPES, feature_columns

CAT_FEATURES = ["task_index"]


def base_features(df: pd.DataFrame) -> list[str]:
    """86 documented feature columns + task_index (categorical). No identifiers."""
    return feature_columns(df.columns) + ["task_index"]


def build(df: pd.DataFrame, groups: tuple[str, ...] = (), drop: tuple[str, ...] = ()) -> tuple[pd.DataFrame, list[str]]:
    """Return (frame with engineered columns added, ordered feature list).

    `groups` selects optional feature groups; every group uses only the rows of the same sample
    (or earlier snapshots of the same session for temporal groups)."""
    assert "target_index" not in groups
    out = df.copy()
    feats = base_features(df)
    for g in groups:
        new = GROUPS[g](out)
        for c, v in new.items():
            out[c] = np.asarray(v, dtype=np.float32)
        feats += list(new.keys())
    feats = [f for f in feats if f not in drop]
    assert not set(feats) & (set(ID_COLS) - {"task_index"}), "identifier leaked into features"
    return out, feats


def aircraft_key(df: pd.DataFrame) -> np.ndarray:
    """Integer id of (sample_id, subject callsign); used only for grouping, never as a feature."""
    names = df["task_name"].astype(str).to_numpy()
    plen = df["task_index"].map(lambda t: len(TASK_TYPES[int(t)]) + 1).to_numpy()
    cs = [n[p:].split("_")[0] for n, p in zip(names, plen)]
    return pd.MultiIndex.from_arrays([df["sample_id"].to_numpy(), cs]).factorize()[0]


def _pct_rank(df: pd.DataFrame, values, by) -> np.ndarray:
    s = pd.Series(np.asarray(values, dtype=np.float64), index=df.index)
    return s.groupby(by).rank(pct=True, method="min").to_numpy()


def rel_features(df: pd.DataFrame) -> dict:
    """Within-sample urgency ranks, distances from the sample minimum, and sample-level type counts."""
    sid = df["sample_id"].to_numpy()
    st = [sid, df["task_index"].to_numpy()]
    abs_tts = df["time_to_sector"].abs()
    cols = {"abs_tts": abs_tts, "time_to_MD": df["time_to_MD"], "distance_to_zone": df["distance_to_zone"],
            "k_timeToExitPoint": df["k_timeToExitPoint"], "x_task_age": df["x_task_age"],
            "x_state_age": df["x_state_age"], "a_since_flight": df["a_since_flight"]}
    out = {}
    for n, v in cols.items():
        out[f"r_s_{n}"] = _pct_rank(df, v, sid)
        if n not in ("abs_tts", "x_task_age"):  # l_rank_absf1_type / l_rank_age_type already exist
            out[f"r_t_{n}"] = _pct_rank(df, v, st)
    for n in ("abs_tts", "x_task_age", "a_since_flight"):
        v = cols[n].astype(np.float64)
        out[f"d_min_{n}"] = (v - v.groupby(sid).transform("min")).to_numpy()
    ti = df["task_index"].to_numpy()
    flags = {"n_transfer": ti == 5, "n_assume": ti == 4, "n_zone": ti == 12, "n_conflict": np.isin(ti, [0, 1, 6]),
             "n_rtr": ti == 13, "n_nonconf": ti == 7,
             "n_xfer_in": (ti == 4) & (df["x_transfer_in"].to_numpy() == 1),
             "n_need_xfer": (ti == 5) & (df["k_aircraftNeedsToBeTransferred"].to_numpy() == 1)}
    for n, f in flags.items():
        out[n] = pd.Series(f.astype(np.float32), index=df.index).groupby(sid).transform("sum").to_numpy()
    return out


def ac_features(df: pd.DataFrame) -> dict:
    """Aircraft-level task-set flags and how this task's age compares with the aircraft's other tasks."""
    ac = aircraft_key(df)
    ti = pd.Series(df["task_index"].to_numpy(), index=df.index)
    out = {}
    for n, types in (("has_entry_coord", [2]), ("has_exit_coord", [3]), ("has_assume", [4]), ("has_transfer", [5]),
                     ("has_conflict", [0, 1, 6]), ("has_exit_cond", [10]), ("has_zone", [12]), ("has_rtr", [13]),
                     ("has_qos", [8])):
        out[f"ac_{n}"] = ti.isin(types).astype(np.float32).groupby(ac).transform("max").to_numpy()
    age = df["x_task_age"].astype(np.float64)
    g = age.groupby(ac)
    out["ac_age_minus_min"] = (age - g.transform("min")).to_numpy()
    out["ac_age_minus_max"] = (age - g.transform("max")).to_numpy()
    out["ac_age_rank"] = g.rank(method="min", ascending=True).to_numpy()
    a_age = age.where(ti == 4).groupby(ac).transform("max")
    e_age = age.where(ti == 2).groupby(ac).transform("max")
    out["twin_age_diff"] = (a_age - e_age).to_numpy()  # NaN unless the aircraft has both tasks
    return out


def seconds(ts: pd.Series) -> np.ndarray:
    p = ts.astype(str).str.split(":", expand=True).astype(int)
    return (p[0] * 3600 + p[1] * 60 + p[2]).to_numpy()


def prev_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Per sample: the previous snapshot of the same session (earlier timestamp) and the time gap.
    Strictly causal: only earlier snapshots are referenced."""
    smp = df.groupby("sample_id", sort=True)[["session", "timestamp"]].first()
    smp["sec"] = seconds(smp["timestamp"])
    smp = smp.sort_values(["session", "sec"], kind="stable")
    same = smp["session"].eq(smp["session"].shift()).fillna(False).astype(bool)
    smp["prev_sid"] = np.where(same, smp.index.to_series().shift().fillna(-1).astype(int), -1)
    smp["gap"] = np.where(same, smp["sec"] - smp["sec"].shift(), np.nan)
    smp["first_sec"] = smp.groupby("session")["sec"].transform("min")
    if (smp["gap"] <= 0).any():
        raise ValueError("timestamps not strictly increasing within session")
    return smp.sort_index()


TEMP_DELTA = ["time_to_sector", "x_task_age", "a_since_flight", "time_to_MD", "x_transfer_in", "is_assumed",
              "conflict_ind", "k_aircraftNeedsToBeTransferred", "x_isc_coordinated", "monitoring_rule_activated",
              "zone_conflict_ind", "x_isc_notified", "x_entry_coord_not_done", "k_out_NOTIFIED", "k_out_COORDINATED",
              "k_outCoordRevision", "k_outCoordMismatch", "k_clearedFL", "k_coordinationToolActivation",
              "k_transferToolActivation", "is_transferred", "exit_FL"]


def temp_features(df: pd.DataFrame) -> dict:
    """Deltas versus the previous snapshot of the same session, aligned by task_name."""
    smp = prev_sample(df)
    prev_sid = df["sample_id"].map(smp["prev_sid"]).to_numpy()
    gap = df["sample_id"].map(smp["gap"]).to_numpy(dtype=np.float64)
    left = pd.DataFrame({"sid": prev_sid, "task_name": df["task_name"].astype(str).to_numpy()})
    right = df[["sample_id", "task_name"] + TEMP_DELTA].copy()
    right["task_name"] = right["task_name"].astype(str)
    right = right.rename(columns={"sample_id": "sid"})
    right["present"] = 1.0
    m = left.merge(right, on=["sid", "task_name"], how="left")
    assert len(m) == len(df)
    has_prev = prev_sid >= 0
    out = {"t_gap": gap, "t_prev_present": np.where(has_prev, m["present"].fillna(0.0).to_numpy(), np.nan)}
    for c in TEMP_DELTA:
        out[f"t_d_{c}"] = df[c].to_numpy(dtype=np.float64) - m[c].to_numpy(dtype=np.float64)
    out["t_new_input"] = np.where(has_prev, (df["a_since_flight"].to_numpy() < gap).astype(float), np.nan)
    a_n60 = df["a_n60"].to_numpy(dtype=np.float64)
    prev_n60 = pd.Series(df.groupby("sample_id")["a_n60"].first()).reindex(prev_sid).to_numpy()
    out["t_d_a_n60"] = a_n60 - prev_n60
    return out


def _flag(df: pd.DataFrame, c: str) -> np.ndarray:
    return (df[c].to_numpy(dtype=np.float64) == 1)


def act_features(df: pd.DataFrame) -> dict:
    """Per-type 'is this task actionable now' flag and the type-relevant urgency (NaN for other types)."""
    ti = df["task_index"].to_numpy()
    ready = np.full(len(df), np.nan)
    rules = {
        5: _flag(df, "k_aircraftNeedsToBeTransferred") | _flag(df, "k_transferToolActivation"),
        3: _flag(df, "k_coordinationToolActivation") | _flag(df, "k_outCoordMismatch"),
        10: _flag(df, "k_hasToClimbToExitFL") | _flag(df, "k_hasToDescendToExitFL"),
        4: _flag(df, "x_transfer_in"),
        2: _flag(df, "x_transfer_in") | _flag(df, "x_isc_coordinated"),
        12: _flag(df, "zone_conflict_ind"),
        13: _flag(df, "monitoring_rule_activated"),
        7: _flag(df, "monitoring_rule_activated"),
    }
    for t in (0, 1, 6):
        rules[t] = _flag(df, "conflict_ind") | _flag(df, "HMI_conflict_safety_alert")
    for t, r in rules.items():
        ready[ti == t] = r[ti == t].astype(float)
    urg_src = {3: "k_timeToExitPoint", 5: "k_timeToExitPoint", 10: "k_timeToExitPoint", 2: "time_to_sector",
               4: "time_to_sector", 12: "distance_to_zone", 0: "time_to_MD", 1: "time_to_MD", 6: "time_to_MD"}
    urg = np.full(len(df), np.nan)
    for t, c in urg_src.items():
        m = ti == t
        urg[m] = np.abs(df[c].to_numpy(dtype=np.float64)[m])
    sid = df["sample_id"].to_numpy()
    out = {"act_ready": ready, "act_urgency": urg,
           "act_urgency_rank_type": _pct_rank(df, urg, [sid, ti]),
           "act_ready_in_type": pd.Series(ready, index=df.index).groupby([sid, ti]).transform("sum").to_numpy()}
    return out


TYPED = {3: ["k_coordinationToolActivation", "k_outCoordMismatch", "k_exitConditionsAreMet", "k_out_NOTIFIED",
             "k_out_COORDINATED", "k_timeToExitPoint", "a_is_last_cs"],
         2: ["x_isc_coordinated", "x_entry_coord_not_done", "x_transfer_in", "time_to_sector", "x_task_age"],
         10: ["k_hasToDescendToExitFL", "k_hasToClimbToExitFL", "k_isAtExitFL", "k_cfl_minus_fl", "k_outCoordMismatch",
              "k_timeToExitPoint"],
         5: ["k_aircraftNeedsToBeTransferred", "k_transferToolActivation", "k_timeToExitPoint"],
         4: ["x_transfer_in", "x_state_age", "time_to_sector"]}


def typed_features(df: pd.DataFrame) -> dict:
    """Type-restricted copies of the signals that separate targets within a type (NaN for other types)."""
    ti = df["task_index"].to_numpy()
    out = {}
    for t, cols in TYPED.items():
        m = ti == t
        for c in cols:
            v = np.full(len(df), np.nan)
            v[m] = df[c].to_numpy(dtype=np.float64)[m]
            out[f"ty{t}_{c}"] = v
    return out


def lead_features(df: pd.DataFrame) -> dict:
    """DIAGNOSTIC ONLY - NON-CAUSAL: deltas versus the NEXT snapshot of the same session (violates ground rule 6).
    Used solely to measure how much future-snapshot information would add; blocked from final refits."""
    smp = df.groupby("sample_id", sort=True)[["session", "timestamp"]].first()
    smp["sec"] = seconds(smp["timestamp"])
    smp = smp.sort_values(["session", "sec"], kind="stable")
    same = smp["session"].eq(smp["session"].shift(-1)).fillna(False).astype(bool)
    smp["next_sid"] = np.where(same, smp.index.to_series().shift(-1).fillna(-1).astype(int), -1)
    smp["gap"] = np.where(same, smp["sec"].shift(-1) - smp["sec"], np.nan)
    smp = smp.sort_index()
    nxt = df["sample_id"].map(smp["next_sid"]).to_numpy()
    gap = df["sample_id"].map(smp["gap"]).to_numpy(dtype=np.float64)
    cols = ["a_since_flight", "x_task_age", "is_assumed", "x_transfer_in", "k_out_COORDINATED", "x_isc_coordinated",
            "k_clearedFL", "k_outCoordRevision", "a_last_assign_level_clearance", "a_last_assign_lateral_clearance",
            "a_last_assume_control", "a_last_initiate_transfer_of_control"]
    left = pd.DataFrame({"sid": nxt, "task_name": df["task_name"].astype(str).to_numpy()})
    right = df[["sample_id", "task_name"] + cols].copy()
    right["task_name"] = right["task_name"].astype(str)
    right = right.rename(columns={"sample_id": "sid"})
    right["present"] = 1.0
    m = left.merge(right, on=["sid", "task_name"], how="left")
    has = nxt >= 0
    out = {"f_gap": gap, "f_present": np.where(has, m["present"].fillna(0.0).to_numpy(), np.nan)}
    for c in cols:
        out[f"f_d_{c}"] = m[c].to_numpy(dtype=np.float64) - df[c].to_numpy(dtype=np.float64)
    out["f_new_input"] = np.where(has & ~np.isnan(m["a_since_flight"].to_numpy(dtype=float)),
                                  (m["a_since_flight"].to_numpy(dtype=float) < gap).astype(float), np.nan)
    return out


# ---------------------------------------------------------------- exit coordination (task_index 3)
# xc: causal (this snapshot and earlier ones). xc_lead: reads LATER snapshots of the same session; a spec that uses
# it must opt in with future=True (see FUTURE_GROUPS and src/train.py).
FUTURE_GROUPS = {"lead", "xc_lead"}
XC_STATE = ["k_outCoordRevision", "k_out_COORDINATED", "k_out_NOTIFIED", "k_outCoordMismatch"]
XC_TRACK = XC_STATE + ["exit_FL", "k_clearedFL"]
INPUT_FLAGS = ["a_last_assume_control", "a_last_force_assume_control", "a_last_assign_level_clearance",
               "a_last_assign_lateral_clearance", "a_last_initiate_transfer_of_control"]
_XC_CACHE: dict = {}


def callsign(df: pd.DataFrame) -> np.ndarray:
    """Subject callsign of each row: the first callsign after the task-type prefix of task_name."""
    names = df["task_name"].astype(str)
    out = pd.Series("", index=df.index, dtype=object)
    ti = df["task_index"].to_numpy()
    for t in np.unique(ti):
        m = ti == t
        out.loc[m] = names[m].str.slice(len(TASK_TYPES[int(t)]) + 1).str.split("_").str[0].to_numpy()
    return out.to_numpy()


def _on(mask: np.ndarray, v) -> np.ndarray:
    out = np.full(len(mask), np.nan)
    v = np.asarray(v, dtype=np.float64)
    out[mask] = v[mask]
    return out


def _xc_context(df: pd.DataFrame) -> dict:
    """Per-row timelines of each aircraft within its session (cached per frame). Keys 'since_*' look only at
    earlier snapshots; keys 'to_*', 'vanish', 'block_end' and 'next_snap' look at later snapshots."""
    sid = df["sample_id"].to_numpy()
    key = (len(df), int(sid[0]), int(sid[-1]), int(sid.astype(np.int64).sum()))
    if key in _XC_CACHE:
        return _XC_CACHE[key]
    sess = df["session"].astype(str).to_numpy()
    cs = callsign(df)
    smp = df.groupby("sample_id", sort=True)["timestamp"].first()
    sec = df["sample_id"].map(pd.Series(seconds(smp), index=smp.index)).to_numpy().astype(np.int64)
    ac = pd.MultiIndex.from_arrays([sess, cs]).factorize()[0]
    out: dict = {}

    # one row per (sample, aircraft), sorted by aircraft then time
    first = ~pd.DataFrame({"sid": sid, "ac": ac}).duplicated().to_numpy()
    tl = pd.DataFrame({"sid": sid[first], "ac": ac[first], "sec": sec[first]})
    for c in XC_TRACK + INPUT_FLAGS + ["a_since_flight"]:
        tl[c] = df[c].to_numpy(dtype=np.float64)[first]
    tl = tl.sort_values(["ac", "sec"], kind="stable").reset_index(drop=True)
    g = tl["ac"]
    same_prev = g.eq(g.shift()).to_numpy()
    prev_sec = tl["sec"].shift().to_numpy(dtype=np.float64)
    secf = tl["sec"].to_numpy(dtype=np.float64)
    obs_gap = np.where(same_prev, secf - prev_sec, np.nan)

    def since_to(chg: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ev = pd.Series(np.where(chg, secf, np.nan))
        last = ev.groupby(g).ffill().to_numpy()
        nxt = ev.groupby(g).shift(-1).groupby(g).bfill().to_numpy()
        nxt_gap = pd.Series(np.where(chg, obs_gap, np.nan)).groupby(g).shift(-1).groupby(g).bfill().to_numpy()
        return secf - last, nxt - secf, nxt_gap

    tlo: dict = {}
    any_chg = np.zeros(len(tl), dtype=bool)
    for c in XC_TRACK:
        v = tl[c].fillna(-999.0)
        chg = same_prev & v.ne(v.shift()).to_numpy()
        if c in XC_STATE:
            any_chg |= chg
        tlo[f"since_{c}"], tlo[f"to_{c}"], _ = since_to(chg)
    tlo["since_any"], tlo["to_any"], tlo["to_any_obs_gap"] = since_to(any_chg)
    tlo["next_snap"] = tl.groupby("ac")["sec"].shift(-1).to_numpy(dtype=np.float64) - secf

    # controller-input events of the aircraft, reconstructed from a_since_flight
    ok = tl["a_since_flight"].to_numpy() < 3600
    flags = np.nan_to_num(tl[INPUT_FLAGS].to_numpy())
    itype = np.where(flags.sum(1) > 0, flags.argmax(1), -1)
    ev = pd.DataFrame({"ac": tl["ac"].to_numpy()[ok],
                       "t_in": np.round(secf[ok] - tl["a_since_flight"].to_numpy()[ok]).astype(np.int64),
                       "itype": itype[ok]}).drop_duplicates(["ac", "t_in"]).sort_values("t_in", kind="stable")
    q = pd.DataFrame({"pos": np.arange(len(tl)), "ac": tl["ac"].to_numpy(), "sec": tl["sec"].to_numpy()})
    q = q.sort_values("sec", kind="stable")
    m = pd.merge_asof(q, ev, left_on="sec", right_on="t_in", by="ac", direction="forward", allow_exact_matches=False)
    m = m.sort_values("pos")
    tlo["to_input"] = m["t_in"].to_numpy(dtype=np.float64) - secf
    it = m["itype"].to_numpy(dtype=np.float64)
    has = np.isfinite(tlo["to_input"])
    tlo["next_in_level"] = np.where(has, (it == 2).astype(float), np.nan)
    tlo["next_in_lateral"] = np.where(has, (it == 3).astype(float), np.nan)
    tlo["next_in_other"] = np.where(has, (~np.isin(it, [2, 3])).astype(float), np.nan)

    # map aircraft-timeline values back to rows
    idx = pd.MultiIndex.from_arrays([tl["sid"].to_numpy(), tl["ac"].to_numpy()]).get_indexer(
        pd.MultiIndex.from_arrays([sid, ac]))
    assert (idx >= 0).all()
    for k, v in tlo.items():
        out[k] = v[idx]

    # task lifetime: time until the (session, task_name) is seen for the last time; NaN if present at session end
    k = pd.DataFrame({"s": sess, "tn": df["task_name"].astype(str).to_numpy(), "sec": sec})
    last_seen = k.groupby(["s", "tn"])["sec"].transform("max").to_numpy()
    sess_end = k.groupby("s")["sec"].transform("max").to_numpy()
    out["vanish"] = np.where(last_seen < sess_end, last_seen - sec, np.nan).astype(np.float64)

    # blocks of consecutive snapshots at most 3 s apart
    b = pd.DataFrame({"s": sess, "sec": sec, "sid": sid}).drop_duplicates("sid").sort_values(["s", "sec"])
    new = ~(b["s"].eq(b["s"].shift()) & (b["sec"] - b["sec"].shift() <= 3)).to_numpy()
    b["blk"] = np.cumsum(new)
    b["end"] = b.groupby("blk")["sec"].transform("max")
    b["start"] = b.groupby("blk")["sec"].transform("min")
    b = b.set_index("sid")
    out["block_end"] = (df["sample_id"].map(b["end"]).to_numpy() - sec).astype(np.float64)
    out["block_start"] = (sec - df["sample_id"].map(b["start"]).to_numpy()).astype(np.float64)
    out["ac_in_sample"] = pd.MultiIndex.from_arrays([sid, cs]).factorize()[0]
    _XC_CACHE.clear()
    _XC_CACHE[key] = out
    return out


def xc_features(df: pd.DataFrame) -> dict:
    """Causal exit-coordination features: type-3 rows only (NaN elsewhere), plus two sample-level counts."""
    ti = df["task_index"].to_numpy()
    is3 = ti == 3
    sid = df["sample_id"].to_numpy()

    def f(c):
        return df[c].to_numpy(dtype=np.float64)

    exit_fl, cfl, fl = f("exit_FL"), f("k_clearedFL"), f("current_FL")
    ctx = _xc_context(df)
    out = {
        "xc_exit_minus_cfl": _on(is3, exit_fl - cfl),
        "xc_abs_exit_minus_cfl": _on(is3, np.abs(exit_fl - cfl)),
        "xc_exit_minus_fl": _on(is3, exit_fl - fl),
        "xc_abs_exit_minus_fl": _on(is3, np.abs(exit_fl - fl)),
        "xc_cfl_ne_exit": _on(is3, np.where(np.isnan(exit_fl) | np.isnan(cfl), np.nan, (exit_fl != cfl).astype(float))),
        "xc_level_clr_age": _on(is3, f("k_levelClearanceAgeS")),
    }
    mm, met = f("k_outCoordMismatch"), f("k_exitConditionsAreMet")
    out["xc_mismatch_met"] = _on(is3, np.where(np.isnan(mm) | np.isnan(met), np.nan, 2 * mm + met))
    out["xc_since_state_chg"] = _on(is3, ctx["since_any"])
    out["xc_since_cfl_chg"] = _on(is3, ctx["since_k_clearedFL"])
    out["xc_since_exitfl_chg"] = _on(is3, ctx["since_exit_FL"])
    for c, n in (("a_since_flight", "asf"), ("k_levelClearanceAgeS", "lca"), ("k_timeToExitPoint", "ttx")):
        out[f"xc_rank3_{n}"] = _pct_rank(df, np.where(is3, f(c), np.nan), sid)
    acs = ctx["ac_in_sample"]
    for n, types in (("has_exit_cond", [10]), ("has_transfer", [5]), ("has_conflict", [0, 1, 6])):
        out[f"xc_{n}"] = _on(is3, pd.Series(np.isin(ti, types).astype(float)).groupby(acs).transform("max").to_numpy())
    out["xc_n3_mismatch"] = pd.Series((is3 & (mm == 1)).astype(float)).groupby(sid).transform("sum").to_numpy()
    ready = ((ti == 4) & (f("x_transfer_in") == 1)) | ((ti == 5) & (f("k_aircraftNeedsToBeTransferred") == 1))
    out["xc_n_ready"] = pd.Series(ready.astype(float)).groupby(sid).transform("sum").to_numpy()
    return out


def xc_lead_features(df: pd.DataFrame) -> dict:
    """FORWARD-LOOKING exit-coordination features (later snapshots of the same session), type-3 rows only."""
    is3 = df["task_index"].to_numpy() == 3
    ctx = _xc_context(df)
    out = {f"xcl_to_chg_{c}": _on(is3, ctx[f"to_{c}"]) for c in XC_TRACK}
    out["xcl_to_state_chg"] = _on(is3, ctx["to_any"])
    out["xcl_state_chg_obs_gap"] = _on(is3, ctx["to_any_obs_gap"])
    out["xcl_next_snap_gap"] = _on(is3, ctx["next_snap"])
    for k in ("to_input", "next_in_level", "next_in_lateral", "next_in_other"):
        out[f"xcl_{k}"] = _on(is3, ctx[k])
    out["xcl_to_vanish"] = _on(is3, ctx["vanish"])
    out["xcl_to_block_end"] = _on(is3, ctx["block_end"])
    out["xcl_vanish_minus_block_end"] = _on(is3, ctx["vanish"] - ctx["block_end"])
    return out


GROUPS: dict = {"lead": lead_features, "rel": rel_features, "ac": ac_features, "temp": temp_features, "act": act_features,
                "typed": typed_features, "xc": xc_features, "xc_lead": xc_lead_features}


# ---------------------------------------------------------------- §8.D flagged experiment: target encoding
# Uses labels, so it lives outside build(): encodings for any row come only from rows of *other*
# controllers (out-of-fold), smoothed toward the training type prior.
TE_KEYS = {"te_task": ["scenario", "task_name"]}


def te_fit(lab: pd.DataFrame, m: float = 20.0) -> dict:
    prior = lab.groupby("task_index")["target_index"].mean()
    tables = {}
    for name, keys in TE_KEYS.items():
        g = lab.groupby([lab[k].astype(str) if k == "task_name" else lab[k] for k in keys])
        agg = pd.DataFrame({"pos": g["target_index"].sum(), "n": g["target_index"].size(),
                            "ti": g["task_index"].first()})
        tables[name] = (agg["pos"] + m * agg["ti"].map(prior)) / (agg["n"] + m)
    return {"prior": prior, "tables": tables}


def te_apply(df: pd.DataFrame, enc: dict) -> dict:
    out = {}
    prior = df["task_index"].map(enc["prior"]).to_numpy(dtype=np.float64)
    for name, keys in TE_KEYS.items():
        idx = pd.MultiIndex.from_arrays([df[k].astype(str) if k == "task_name" else df[k] for k in keys])
        v = enc["tables"][name].reindex(idx).to_numpy(dtype=np.float64)
        v = np.where(np.isnan(v), prior, v)
        out[name] = v
        out[f"{name}_ratio"] = v / np.maximum(prior, 1e-6)
        out[f"{name}_rs"] = _pct_rank(df, v, df["sample_id"].to_numpy())
    return out


def te_oof(lab: pd.DataFrame, groups: np.ndarray, m: float = 20.0) -> dict:
    """Out-of-group encodings for labelled rows: each group encoded from all other groups."""
    out: dict = {}
    for gval in np.unique(groups):
        msk = groups == gval
        part = te_apply(lab[msk], te_fit(lab[~msk], m))
        for c, v in part.items():
            out.setdefault(c, np.full(len(lab), np.nan))[msk] = v
    return out


def add_cols(df: pd.DataFrame, cols: dict) -> pd.DataFrame:
    df = df.copy()
    for c, v in cols.items():
        df[c] = np.asarray(v, dtype=np.float32)
    return df


# ---------------------------------------------------------------- NON-CAUSAL (future snapshots), opt-in only
# Uses later snapshots of the same session, which test.csv contains in full. Not a real-time model.
def blocks(df: pd.DataFrame, max_gap: float = 3.0) -> pd.DataFrame:
    """Per sample: block of consecutive snapshots (gap <= max_gap s) within a session, and its time span."""
    smp = df.groupby("sample_id", sort=True)[["session", "timestamp"]].first()
    smp["T"] = seconds(smp["timestamp"]).astype(float)
    smp = smp.sort_values(["session", "T"], kind="stable")
    sv = smp["session"].astype(str).to_numpy()
    tv = smp["T"].to_numpy()
    new = np.r_[True, (sv[1:] != sv[:-1]) | (np.diff(tv) > max_gap)]
    smp["block"] = np.cumsum(new)
    g = smp.groupby("block")["T"]
    smp["b_start"], smp["b_end"], smp["b_n"] = g.transform("min"), g.transform("max"), g.transform("size")
    smp["b_pos"] = smp.groupby("block").cumcount()
    smp["session"] = smp["session"].astype(str)
    nxt = smp.groupby("session")["T"].shift(-1)
    smp["next_T"] = nxt
    blk_next = smp.groupby("block")["next_T"].transform("last")  # first snapshot after the block
    smp["b_next_start"] = blk_next
    return smp.sort_index()


def lead_blk_features(df: pd.DataFrame) -> dict:
    """Block position, task lifetime relative to the block, and controller inputs on the aircraft around the block."""
    smp = blocks(df)
    sid = df["sample_id"]
    T = sid.map(smp["T"]).to_numpy()
    b_start, b_end = sid.map(smp["b_start"]).to_numpy(), sid.map(smp["b_end"]).to_numpy()
    b_n, b_next = sid.map(smp["b_n"]).to_numpy(dtype=float), sid.map(smp["b_next_start"]).to_numpy(dtype=float)
    block = sid.map(smp["block"]).to_numpy()
    sess = df["session"].astype(str).to_numpy()
    name = df["task_name"].astype(str).to_numpy()
    key = pd.MultiIndex.from_arrays([sess, name]).factorize()[0]
    Ts = pd.Series(T)
    first_T = Ts.groupby(key).transform("min").to_numpy()
    last_T = Ts.groupby(key).transform("max").to_numpy()
    sess_last = Ts.groupby(sess).transform("max").to_numpy()
    alive_end = last_T >= sess_last
    # first snapshot of the session after the task's last sighting (the task is gone there)
    smp_s = smp.sort_values(["session", "T"])
    nxt_map = dict(zip(zip(smp_s["session"].astype(str), smp_s["T"]), smp_s["next_T"]))
    gone_T = np.array([nxt_map.get((s, t), np.nan) for s, t in zip(sess, last_T)], dtype=float)
    out = {
        "L_b_n": b_n, "L_b_pos_frac": (T - b_start) / np.maximum(b_end - b_start, 1.0),
        "L_t_to_b_end": b_end - T, "L_t_from_b_start": T - b_start, "L_gap_to_next_block": b_next - b_end,
        "L_alive_at_end": alive_end.astype(float),
        "L_last_minus_b_end": np.where(alive_end, np.nan, last_T - b_end),
        "L_gone_minus_b_end": np.where(alive_end, np.nan, gone_T - b_end),
        "L_last_minus_T": np.where(alive_end, np.nan, last_T - T),
        "L_first_minus_b_start": first_T - b_start,
        "L_gone_in_window": np.where(alive_end, 0.0, ((last_T >= b_end) & (last_T <= b_end) ).astype(float)),
    }
    # share of the block's snapshots that contain this task
    bk = pd.Series(np.ones(len(df)), index=df.index).groupby([block, key]).transform("sum").to_numpy()
    out["L_block_presence"] = bk / b_n
    # controller inputs on the aircraft: event time = T - a_since_flight, observed at any snapshot
    ac = aircraft_key(df)
    cs = pd.Series(df["task_name"].astype(str).to_numpy())
    plen = df["task_index"].map(lambda t: len(TASK_TYPES[int(t)]) + 1).to_numpy()
    callsign = np.array([n[p:].split("_")[0] for n, p in zip(name, plen)])
    a = df["a_since_flight"].to_numpy(dtype=float)
    ev = pd.DataFrame({"s": sess, "cs": callsign, "t": np.round(T - a), "lvl": df["a_last_assign_level_clearance"].to_numpy(),
                       "lat": df["a_last_assign_lateral_clearance"].to_numpy(), "asm": df["a_last_assume_control"].to_numpy(),
                       "xfer": df["a_last_initiate_transfer_of_control"].to_numpy()})[a < 3600]
    ev = ev.drop_duplicates(["s", "cs", "t"]).sort_values("t")
    q = pd.DataFrame({"s": sess, "cs": callsign, "q": b_start - 3.0, "i": np.arange(len(df))}).sort_values("q")
    m = pd.merge_asof(q, ev.rename(columns={"t": "q_ev"}).assign(q=lambda x: x["q_ev"]).sort_values("q"),
                      on="q", by=["s", "cs"], direction="forward", allow_exact_matches=True)
    m = m.sort_values("i")
    t_next = m["q_ev"].to_numpy(dtype=float)
    out["L_next_input_minus_b_start"] = t_next - b_start
    out["L_next_input_minus_b_end"] = t_next - b_end
    out["L_input_in_block"] = ((t_next >= b_start - 3) & (t_next <= b_end + 3)).astype(float)
    for c in ("lvl", "lat", "asm", "xfer"):
        out[f"L_next_input_{c}"] = m[c].to_numpy(dtype=float)
    return out


GROUPS["lead_blk"] = lead_blk_features

FUTURE_GROUPS.add("lead_blk")


CHG_COLS = ["is_assumed", "x_transfer_in", "x_isc_notified", "x_isc_coordinated", "x_entry_coord_not_done",
            "k_out_NOTIFIED", "k_out_COORDINATED", "k_outCoordRevision", "k_outCoordMismatch", "k_clearedFL", "exit_FL",
            "k_exitConditionsAreMet", "k_isClearedHeadingAssigned", "monitoring_rule_activated", "conflict_ind",
            "zone_conflict_ind", "qos_candidate", "k_coordinationToolActivation", "k_transferToolActivation",
            "non_conformance_agreed_route_deviation_horizontally"]


def lead_chg_features(df: pd.DataFrame) -> dict:
    """NON-CAUSAL: aircraft state just after the block (next snapshot of the aircraft) and just before it,
    compared with the current snapshot, plus whether this task still exists right after the block."""
    smp = blocks(df)
    sid = df["sample_id"]
    T = sid.map(smp["T"]).to_numpy(dtype=float)
    b_start = sid.map(smp["b_start"]).to_numpy(dtype=float)
    b_end = sid.map(smp["b_end"]).to_numpy(dtype=float)
    sess = df["session"].astype(str).to_numpy()
    name = df["task_name"].astype(str).to_numpy()
    plen = df["task_index"].map(lambda t: len(TASK_TYPES[int(t)]) + 1).to_numpy()
    cs = np.array([n[p:].split("_")[0] for n, p in zip(name, plen)])
    # aircraft-level timeline: one row per (session, callsign, T)
    tl = pd.DataFrame({"s": sess, "cs": cs, "t": T})
    for c in CHG_COLS:
        tl[c] = df[c].to_numpy(dtype=float)
    tl = tl.drop_duplicates(["s", "cs", "t"]).sort_values("t")
    cur = tl.set_index(["s", "cs", "t"])
    q = pd.DataFrame({"s": sess, "cs": cs, "i": np.arange(len(df))})
    out = {}
    for side, qt, direction in (("next", b_end + 0.5, "forward"), ("prev", b_start - 0.5, "backward")):
        qq = q.assign(t=qt).sort_values("t")
        m = pd.merge_asof(qq, tl.rename(columns={"t": "t_hit"}).assign(t=lambda x: x["t_hit"]).sort_values("t"),
                          on="t", by=["s", "cs"], direction=direction).sort_values("i")
        dt = (m["t_hit"].to_numpy() - b_end) if side == "next" else (b_start - m["t_hit"].to_numpy())
        out[f"C_{side}_dt"] = dt
        n_changed = np.zeros(len(df))
        for c in CHG_COLS:
            v_now = df[c].to_numpy(dtype=float)
            v_o = m[c].to_numpy(dtype=float)
            d = (v_o - v_now) if side == "next" else (v_now - v_o)
            out[f"C_{side}_d_{c}"] = d
            n_changed += np.nan_to_num(np.abs(d) > 0)
        out[f"C_{side}_n_changed"] = np.where(np.isnan(dt), np.nan, n_changed)
    # does this exact task exist at the aircraft's next snapshot after the block?
    tk = pd.DataFrame({"s": sess, "k": name, "t": T}).drop_duplicates().sort_values("t")
    qq = pd.DataFrame({"s": sess, "k": name, "t": b_end + 0.5, "i": np.arange(len(df))}).sort_values("t")
    m = pd.merge_asof(qq, tk.assign(t_task=tk["t"]), on="t", by=["s", "k"], direction="forward").sort_values("i")
    t_task_next = m["t_task"].to_numpy()
    nxt_ac = out["C_next_dt"] + b_end
    out["C_task_present_after"] = np.where(np.isnan(nxt_ac), np.nan, (t_task_next == nxt_ac).astype(float))
    return out


GROUPS["lead_chg"] = lead_chg_features
FUTURE_GROUPS.add("lead_chg")
