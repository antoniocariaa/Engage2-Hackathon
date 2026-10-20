"""Feature engineering versionato (M3).

Ogni versione vN = v(N-1) + una famiglia di feature. Funzioni pure: ricevono il DataFrame grezzo,
non leggono mai target_index, non imputano i NaN.

    v0  feature grezze (87) con task_index categorico
    v1  + scomposizione di time_to_sector
    v2  + contesto di snapshot (rank, pct_rank, z-score, distanza dal minimo)
    v3  + carico dello snapshot
    v4  + aereo ricostruito da task_name e feature intra-aereo
    v5  + storia azioni ATCO rispetto all'apertura del task

Oltre a vN si possono combinare le famiglie per nome, es. "tts+ac+hist" (vedi FAMILIES).
"""
import numpy as np
import pandas as pd

from src.config import DROP_COLS

TASK_TYPES = {
    0: "entry_conflict_resolution", 1: "exit_conflict_resolution", 2: "entry_coordination",
    3: "exit_coordination", 4: "assume_aircraft", 5: "transfer_aircraft", 6: "conflict_resolution",
    7: "non_conformance_resolution", 8: "qos_improvement", 9: "entry_conditions", 10: "exit_conditions",
    11: "unknown_11", 12: "zone_conflict", 13: "return_to_route",
}
CONFLICT_TYPES = (0, 1, 6)
CAT_FEATURES = ["task_index"]
F32 = np.float32


def _raw_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in DROP_COLS]


def _snap(df):
    return df.groupby("sample_id", sort=False)


def add_tts_split(df: pd.DataFrame) -> list:
    tts = df["time_to_sector"]
    df["abs_tts"] = tts.abs().astype(F32)
    df["is_inside"] = (tts < 0).astype(F32).where(tts.notna())
    df["sec_to_entry"] = tts.where(tts >= 0).astype(F32)
    df["sec_to_exit"] = (-tts).where(tts < 0).astype(F32)
    df["abs_dist_sector"] = df["distance_to_sector"].abs().astype(F32)
    return ["abs_tts", "is_inside", "sec_to_entry", "sec_to_exit", "abs_dist_sector"]


SNAP_COLS = ["abs_tts", "sec_to_entry", "sec_to_exit", "abs_dist_sector", "x_task_age", "x_state_age",
             "k_timeToExitPoint", "time_to_MD", "distance_to_MD"]


def add_snapshot_context(df: pd.DataFrame) -> list:
    g = _snap(df)
    n = df["seq_len"].astype(F32)
    new = []
    for c in SNAP_COLS:
        rank = g[c].rank(method="min", ascending=True)
        mean, std = g[c].transform("mean"), g[c].transform("std")
        df[f"{c}_rank"] = rank.astype(F32)
        df[f"{c}_pct"] = (rank / n).astype(F32)
        df[f"{c}_z"] = ((df[c] - mean) / (std + 1e-3)).astype(F32)
        df[f"{c}_dmin"] = (df[c] - g[c].transform("min")).astype(F32)
        new += [f"{c}_rank", f"{c}_pct", f"{c}_z", f"{c}_dmin"]
    # rango dentro lo stesso tipo di task nello snapshot
    gt = df.groupby(["sample_id", "task_index"], sort=False)
    for c in ["abs_tts", "x_task_age"]:
        df[f"{c}_rank_in_type"] = gt[c].rank(method="min").astype(F32)
        new.append(f"{c}_rank_in_type")
    return new


def add_snapshot_load(df: pd.DataFrame) -> list:
    g = _snap(df)
    df["snap_n_type"] = df.groupby(["sample_id", "task_index"], sort=False)["row_index"].transform("size").astype(F32)
    df["snap_frac_type"] = (df["snap_n_type"] / df["seq_len"]).astype(F32)
    df["_crit"] = (df["abs_tts"] < 120).astype(F32)
    df["snap_n_critical"] = g["_crit"].transform("sum").astype(F32)
    df["snap_n_conflict_rows"] = g["conflict_ind"].transform("sum").astype(F32)
    df["snap_n_stca"] = g["HMI_conflict_safety_alert"].transform("sum").astype(F32)
    df["snap_n_inside"] = g["is_inside"].transform("sum").astype(F32)
    df["snap_a_n60_max"] = g["a_n60"].transform("max").astype(F32)
    df.drop(columns="_crit", inplace=True)
    return ["snap_n_type", "snap_frac_type", "snap_n_critical", "snap_n_conflict_rows", "snap_n_stca",
            "snap_n_inside", "snap_a_n60_max"]


def parse_callsigns(task_name: pd.Series, task_index: pd.Series) -> pd.DataFrame:
    """task_name = <tipo>_<cs1>[_<cs2>]; il prefisso è noto da task_index."""
    prefix_len = task_index.map({k: len(v) + 1 for k, v in TASK_TYPES.items()})
    rest = pd.Series([s[p:] for s, p in zip(task_name, prefix_len)], index=task_name.index)
    parts = rest.str.split("_", n=1, expand=True)
    cs2 = parts[1] if 1 in parts.columns else pd.Series(None, index=task_name.index)
    return pd.DataFrame({"callsign": parts[0], "callsign2": cs2})


def add_aircraft(df: pd.DataFrame) -> list:
    cs = parse_callsigns(df["task_name"], df["task_index"])
    df["callsign"] = cs["callsign"].values
    ga = df.groupby(["sample_id", "callsign"], sort=False)
    new = []
    df["ac_n_tasks"] = ga["row_index"].transform("size").astype(F32)
    new.append("ac_n_tasks")
    # tipi presenti sullo stesso aereo nello snapshot
    for k, name in [(2, "entry_coord"), (3, "exit_coord"), (4, "assume"), (5, "transfer"), (12, "zone_conflict"),
                    (7, "non_conf")]:
        col = f"ac_has_{name}"
        df["_t"] = (df["task_index"] == k).astype(F32)
        df[col] = ga["_t"].transform("max").astype(F32)
        new.append(col)
    df["_t"] = df["task_index"].isin(CONFLICT_TYPES).astype(F32)
    df["ac_has_conflict_res"] = ga["_t"].transform("max").astype(F32)
    new.append("ac_has_conflict_res")
    df.drop(columns="_t", inplace=True)
    for c in ["x_task_age", "x_state_age"]:
        mn, mx = ga[c].transform("min"), ga[c].transform("max")
        df[f"{c}_ac_dmin"] = (df[c] - mn).astype(F32)
        df[f"{c}_ac_dmax"] = (df[c] - mx).astype(F32)
        df[f"{c}_ac_ratio"] = (df[c] / (mx + 1.0)).astype(F32)
        df[f"{c}_ac_rank"] = ga[c].rank(method="min").astype(F32)
        new += [f"{c}_ac_dmin", f"{c}_ac_dmax", f"{c}_ac_ratio", f"{c}_ac_rank"]
    # coppie canoniche
    df["pair_assume_entry"] = (df["ac_has_assume"] * df["ac_has_entry_coord"]).astype(F32)
    df["pair_transfer_exit"] = (df["ac_has_transfer"] * df["ac_has_exit_coord"]).astype(F32)
    # quante volte l'aereo compare anche come secondo callsign di un conflitto
    sec = pd.DataFrame({"sample_id": df["sample_id"].values, "callsign": cs["callsign2"].values}).dropna()
    n_sec = sec.groupby(["sample_id", "callsign"]).size().rename("ac_n_as_cs2")
    df["ac_n_as_cs2"] = (df[["sample_id", "callsign"]].merge(n_sec, how="left", on=["sample_id", "callsign"])
                         ["ac_n_as_cs2"].fillna(0).values.astype(F32))
    df["snap_n_aircraft"] = _snap(df)["callsign"].transform("nunique").astype(F32)
    new += ["pair_assume_entry", "pair_transfer_exit", "ac_n_as_cs2", "snap_n_aircraft"]
    return new


def add_atco_history(df: pd.DataFrame) -> list:
    g = _snap(df)
    # il controllore ha toccato l'aereo dopo l'apertura del task?
    df["acted_since_open"] = (df["a_since_flight"] < df["x_task_age"]).astype(F32).where(df["a_since_flight"].notna())
    df["a_since_flight_minus_age"] = (df["a_since_flight"] - df["x_task_age"]).astype(F32)
    df["a_since_flight_rank"] = g["a_since_flight"].rank(method="min").astype(F32)
    df["a_n60_rank"] = g["a_n60"].rank(method="min", ascending=False).astype(F32)
    df["last_cs_x_type"] = (df["a_is_last_cs"] * 100 + df["task_index"]).astype(F32)
    new = ["acted_since_open", "a_since_flight_minus_age", "a_since_flight_rank", "a_n60_rank", "last_cs_x_type"]
    for c in ["a_last_assume_control", "a_last_initiate_transfer_of_control", "a_last_assign_level_clearance"]:
        df[f"{c}_minus_age"] = (df[c] - df["x_task_age"]).astype(F32)
        new.append(f"{c}_minus_age")
    return new


FAMILIES = {"tts": add_tts_split, "snap": add_snapshot_context, "load": add_snapshot_load,
            "ac": add_aircraft, "hist": add_atco_history}
# snap e load usano colonne create da tts
DEPENDS = {"snap": ["tts"], "load": ["tts"]}
VERSIONS = {f"v{i}": list(FAMILIES)[:i] for i in range(len(FAMILIES) + 1)}


def resolve(version: str) -> list:
    """'v3' -> famiglie cumulative; 'tts+ac+hist' -> famiglie scelte (+ dipendenze), sempre nell'ordine di FAMILIES."""
    fams = VERSIONS[version] if version in VERSIONS else version.split("+")
    unknown = set(fams) - set(FAMILIES)
    if unknown:
        raise ValueError(f"famiglie sconosciute {unknown}; usa {list(VERSIONS)} o combinazioni di {list(FAMILIES)}")
    fams = set(fams) | {d for f in fams for d in DEPENDS.get(f, [])}
    return [f for f in FAMILIES if f in fams]


def build_features(df: pd.DataFrame, version: str) -> tuple[pd.DataFrame, list]:
    """Ritorna (df con feature aggiunte, lista feature). Il df deve essere ordinato per sample_id."""
    out = df.drop(columns=["target_index"], errors="ignore").copy()
    feats = _raw_cols(out)
    for fam in resolve(version):
        feats += FAMILIES[fam](out)
    if "target_index" in df.columns:
        out["target_index"] = df["target_index"].values
    out["task_index"] = out["task_index"].astype("category")
    return out, feats
