"""Step 2 of EXIT_COORDINATION_PROMPT.md: a sample-level model of WHICH task type is active, combined with a row ranker.

combined score = z(ranker score within sample) + lam * log P_A(type of the row | sample)

  python -m src.stage_a cv <ranker_experiment> [--future]        # CV (OOF by controller) + validation, logs, npz
  python -m src.stage_a test <ranker_experiment> --lam L [--future] [--out submissions/submission.csv]
Stage A is trained on the same controller folds as the ranker, so its CV probabilities are out-of-fold.
--future adds per-type aggregates of forward-looking signals (task disappearance, next controller input).
"""
from __future__ import annotations

import argparse
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from src import features
from src.cv import controller_folds, fold_array
from src.data_io import ROOT, load_all, load_config
from src.explog import log_experiment
from src.metrics import hits_from_scores
from src.rank import topk, within_sample_z, write_submission

PRED = ROOT / "cache" / "preds"
CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13]
CIDX = {t: i for i, t in enumerate(CLASSES)}
LAMS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
PARAMS = dict(objective="multiclass", num_class=len(CLASSES), learning_rate=0.05, num_leaves=15, min_data_in_leaf=50,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1,
              deterministic=True, force_row_wise=True, num_threads=3)


def sample_features(df: pd.DataFrame, future: bool) -> pd.DataFrame:
    """One row per sample_id (sorted): per-type counts and aggregates of urgency/attention signals."""
    ti = df["task_index"].to_numpy()

    def f(c):
        return df[c].to_numpy(dtype=np.float64)

    ready = (((ti == 4) & (f("x_transfer_in") == 1)) | ((ti == 5) & (f("k_aircraftNeedsToBeTransferred") == 1))
             | (np.isin(ti, [0, 1, 6]) & (f("HMI_conflict_safety_alert") == 1)) | ((ti == 3) & (f("k_outCoordMismatch") == 1))
             | ((ti == 10) & ((f("k_hasToClimbToExitFL") == 1) | (f("k_hasToDescendToExitFL") == 1))))
    b = pd.DataFrame({"sid": df["sample_id"].to_numpy(), "ti": ti, "asf": f("a_since_flight"), "last": f("a_is_last_cs"),
                      "age": f("x_task_age"), "att": np.abs(f("time_to_sector")), "ttx": f("k_timeToExitPoint"),
                      "ready": ready.astype(float), "lvl": f("a_last_assign_level_clearance"),
                      "lat": f("a_last_assign_lateral_clearance")})
    aggs = dict(n=("asf", "size"), asf_min=("asf", "min"), last_max=("last", "max"), age_min=("age", "min"),
                age_max=("age", "max"), att_min=("att", "min"), ttx_min=("ttx", "min"), ready_sum=("ready", "sum"),
                lvl_max=("lvl", "max"), lat_max=("lat", "max"))
    if future:
        ctx = features._xc_context(df)
        b["van"], b["tin"] = ctx["vanish"], ctx["to_input"]
        b["chg"] = ctx["to_any"]
        aggs.update(van_min=("van", "min"), tin_min=("tin", "min"), chg_min=("chg", "min"))
    agg = b.groupby(["sid", "ti"]).agg(**aggs).unstack("ti")
    agg.columns = [f"{a}_t{t}" for a, t in agg.columns]
    for c in [c for c in agg.columns if c.startswith(("n_", "ready_sum_"))]:
        agg[c] = agg[c].fillna(0.0)
    smp = df.groupby("sample_id", sort=True)[["a_n60", "a_n300", "a_since_any", "seq_len"]].first().astype(float)
    out = smp.join(agg).sort_index()
    return out


def align(X: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return X.reindex(columns=cols).to_numpy(dtype=np.float32)


def labels(df: pd.DataFrame) -> pd.Series:
    t = df.loc[df["target_index"] == 1].set_index("sample_id")["task_index"].sort_index()
    return t.map(CIDX)


def fit(X: np.ndarray, y: np.ndarray, rounds: int, seed: int, valid=None):
    curve: dict = {}
    dtr = lgb.Dataset(X, y.astype(int))
    kw = {}
    if valid is not None:
        kw = dict(valid_sets=[lgb.Dataset(valid[0], valid[1].astype(int), reference=dtr)], valid_names=["v"],
                  callbacks=[lgb.record_evaluation(curve), lgb.early_stopping(50, verbose=False)])
    b = lgb.train(dict(PARAMS, seed=seed, metric="multi_logloss"), dtr, num_boost_round=rounds, **kw)
    return b, curve.get("v", {}).get("multi_logloss", [])


def row_logp(df: pd.DataFrame, P: np.ndarray, sids: np.ndarray) -> np.ndarray:
    """log P_A(class of the row's type) for every row."""
    pos = pd.Series(np.arange(len(sids)), index=sids).reindex(df["sample_id"].to_numpy()).to_numpy()
    cls = df["task_index"].map(CIDX).to_numpy()
    return np.log(np.clip(P[pos, cls], 1e-6, 1.0))


def combine(df: pd.DataFrame, ranker_score: np.ndarray, logp: np.ndarray, lam: float) -> np.ndarray:
    return within_sample_z(df, ranker_score) + lam * logp


def run_cv(ranker: str, future: bool, seeds=(42, 43, 44)) -> None:
    cfg = load_config()
    d = load_all(cfg)
    tr, va = d["train"], d["validation"]
    folds = fold_array(tr, controller_folds(tr, cfg["n_folds"]))
    sfold = pd.Series(folds, index=tr["sample_id"].to_numpy()).groupby(level=0).first().sort_index()
    Xtr, Xva = sample_features(tr, future), sample_features(va, future)
    cols = list(Xtr.columns)
    ytr, yva = labels(tr).reindex(Xtr.index).to_numpy(), labels(va).reindex(Xva.index).to_numpy()
    A, Atr = align(Xtr, cols), None
    # pass 1: logloss curves on held-out folds -> common round count R
    curves = []
    for k in range(cfg["n_folds"]):
        m = sfold.to_numpy() == k
        _, c = fit(A[~m], ytr[~m], 1500, seeds[0], valid=(A[m], ytr[m]))
        curves.append(c)
    mean = np.mean([c[:min(map(len, curves))] for c in curves], axis=0)
    R = int(np.argmin(mean)) + 1
    # pass 2: OOF probabilities at R, averaged over seeds
    P = np.zeros((len(A), len(CLASSES)))
    for k in range(cfg["n_folds"]):
        m = sfold.to_numpy() == k
        for s in seeds:
            b, _ = fit(A[~m], ytr[~m], R, s)
            P[m] += b.predict(A[m]) / len(seeds)
    R_full = int(round(R * len(A) / np.mean([(sfold.to_numpy() != k).sum() for k in range(cfg["n_folds"])])))
    Pv = np.zeros((len(Xva), len(CLASSES)))
    for s in seeds:
        b, _ = fit(A, ytr, R_full, s)
        Pv += b.predict(align(Xva, cols)) / len(seeds)
    acc = lambda P_, y_: (np.argsort(-P_, axis=1)[:, :3] == y_[:, None]).any(1).mean()
    print(f"stage A: R={R} R_full={R_full} | type top-1 CV {np.mean(P.argmax(1) == ytr):.3f} val "
          f"{np.mean(Pv.argmax(1) == yva):.3f} | type top-3 CV {acc(P, ytr):.3f} val {acc(Pv, yva):.3f}")
    rk = np.load(PRED / f"{ranker}.npz")
    lp_tr, lp_va = row_logp(tr, P, Xtr.index.to_numpy()), row_logp(va, Pv, Xva.index.to_numpy())

    def fold_scores(s):
        h = hits_from_scores(tr, s)
        return [float(h[sfold.reindex(h.index).to_numpy() == k].mean()) for k in range(cfg["n_folds"])]

    grid = {lam: fold_scores(combine(tr, rk["oof"], lp_tr, lam)) for lam in LAMS}
    vgrid = {lam: float(hits_from_scores(va, combine(va, rk["val"], lp_va, lam)).mean()) for lam in LAMS}
    for lam in LAMS:
        print(f"  lam {lam:<5}: CV {np.mean(grid[lam]):.4f} {[round(x, 4) for x in grid[lam]]} | val {vgrid[lam]:.4f}")
    # nested choice of lam: chosen on the other 3 folds, scored on the held-out fold
    nested = []
    for k in range(cfg["n_folds"]):
        best = max(LAMS, key=lambda l_: np.mean([grid[l_][j] for j in range(cfg["n_folds"]) if j != k]))
        nested.append(grid[best][k])
    lam = max(LAMS, key=lambda l_: np.mean(grid[l_]))
    print(f"  selected lam {lam} (CV over all folds); nested-lam CV {np.mean(nested):.4f} {[round(x, 4) for x in nested]}")
    name = f"{ranker}+typeA" + ("F" if future else "")
    np.savez(PRED / f"{name}.npz", oof=combine(tr, rk["oof"], lp_tr, lam), val=combine(va, rk["val"], lp_va, lam),
             lam=lam, R=R, R_full=R_full)
    log_experiment("xc2", name, dict(ranker=ranker, future=future, lam=lam, R=R, params=PARAMS), grid[lam],
                   vgrid[lam], notes=f"stage A + ranker; lam from CV; nested-lam CV {np.mean(nested):.4f}", R=R,
                   R_full=R_full)
    (ROOT / "artifacts").mkdir(exist_ok=True)
    (ROOT / "artifacts" / f"{name}.json").write_text(json.dumps(dict(ranker=ranker, future=future, lam=lam, R=R,
                                                                     R_full=R_full, cols=cols)))
    print(f"saved {name}")


def run_test(ranker: str, future: bool, lam: float, out: str, seeds=(42, 43, 44)) -> None:
    from src.predict import score_artifact
    from src.validate_submission import validate
    cfg = load_config()
    d = load_all(cfg)
    full = pd.concat([d["train"], d["validation"]], ignore_index=True).sort_values(["sample_id", "row_index"],
                                                                                    kind="stable").reset_index(drop=True)
    te = d["test"]
    meta = json.loads((ROOT / "artifacts" / f"{ranker}+typeA{'F' if future else ''}.json").read_text())
    Xf, Xte = sample_features(full, future), sample_features(te, future)
    cols = meta["cols"]
    yf = labels(full).reindex(Xf.index).to_numpy()
    R_final = int(round(meta["R_full"] * len(Xf) / d["train"]["sample_id"].nunique()))
    P = np.zeros((len(Xte), len(CLASSES)))
    for s in seeds:
        b, _ = fit(align(Xf, cols), yf, R_final, s)
        P += b.predict(align(Xte, cols)) / len(seeds)
    sc = combine(te, score_artifact(ranker, te), row_logp(te, P, Xte.index.to_numpy()), lam)
    path = ROOT / out
    write_submission(topk(te, sc), path)
    validate(path, "test")
    log_experiment("xc2", f"final:{ranker}+typeA{'F' if future else ''}", dict(ranker=ranker, lam=lam, R_final=R_final),
                   notes=f"stage A refit on train+validation, R_final={R_final}; wrote {out}")
    print(f"wrote {out} (lam={lam}, R_final={R_final})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cv", "test"])
    ap.add_argument("ranker")
    ap.add_argument("--future", action="store_true")
    ap.add_argument("--lam", type=float, default=None)
    ap.add_argument("--out", default="submissions/submission.csv")
    a = ap.parse_args()
    if a.cmd == "cv":
        run_cv(a.ranker, a.future)
    else:
        run_test(a.ranker, a.future, a.lam, a.out)


if __name__ == "__main__":
    main()
