"""CV experiments and final fit.  Logs every run to reports/experiments.csv.

  python -m src.train baselines
  python -m src.train cv <experiment> [<experiment> ...]
  python -m src.train final <experiment>
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

from src import baselines, features, models
from src.cv import controller_folds, fold_array, fold_report
from src.data_io import ROOT, load_all, load_config
from src.explog import log_experiment
from src.metrics import hits_from_scores, session_bootstrap

PRED_DIR = ROOT / "cache" / "preds"

# name -> spec. "groups" are optional feature groups from features.GROUPS.
EXPERIMENTS: dict[str, dict] = {
    "lgb_binary": dict(objective="binary"),
    "lgb_binary_w": dict(objective="binary", neg_weight="inv_seq_m1", neg_weight_scale=30.0),
    "lgb_lambdarank": dict(objective="lambdarank"),
    "lgb_xendcg": dict(objective="rank_xendcg"),
    "lgb_softmax": dict(objective="softmax"),
    # Stage 5: within-sample relative (rel) and aircraft-level (ac) features
    "lgb_binary_rel": dict(objective="binary", groups=["rel"]),
    "lgb_binary_ac": dict(objective="binary", groups=["ac"]),
    "lgb_binary_rel_ac": dict(objective="binary", groups=["rel", "ac"]),
    # Stage 6: causal temporal deltas vs the previous snapshot
    "lgb_binary_temp": dict(objective="binary", groups=["temp"]),
    # Stage 4 (optional libraries)
    "xgb_pairwise": dict(lib="xgb", objective="rank:pairwise"),
    # regularisation grid (base features, binary)
    "lgb_binary_l15": dict(objective="binary", params=dict(num_leaves=15, min_data_in_leaf=500)),
    "lgb_binary_l7": dict(objective="binary", params=dict(num_leaves=7, min_data_in_leaf=1000)),
    "lgb_binary_ff5": dict(objective="binary", params=dict(feature_fraction=0.5, lambda_l2=10.0)),
    "lgb_binary_l63": dict(objective="binary", params=dict(num_leaves=63, min_data_in_leaf=100)),
    "lgb_binary_et": dict(objective="binary", params=dict(extra_trees=True)),
    "lgb_binary_l15_ff5": dict(objective="binary", params=dict(num_leaves=15, min_data_in_leaf=500,
                                                                 feature_fraction=0.5, lambda_l2=10.0)),
    # seed variance of the reference model
    "lgb_binary_s43": dict(objective="binary", seeds=[43]),
    "lgb_binary_s44": dict(objective="binary", seeds=[44]),
    # 3-seed versions (score-averaged) for less noisy comparisons
    "lgb_binary_3s": dict(objective="binary", seeds=[42, 43, 44]),
    "lgb_lambdarank_3s": dict(objective="lambdarank", seeds=[42, 43, 44]),
    "lgb_binary_noctx_3s": dict(objective="binary", seeds=[42, 43, 44], drop=["a_since_any", "a_n60", "a_n300"]),
    # lambdarank tuning (3 seeds each; reference lgb_lambdarank_3s)
    "lr_lr02_3s": dict(objective="lambdarank", seeds=[42, 43, 44], params=dict(learning_rate=0.02)),
    "lr_trunc30_3s": dict(objective="lambdarank", seeds=[42, 43, 44], params=dict(lambdarank_truncation_level=30)),
    "lr_md500_3s": dict(objective="lambdarank", seeds=[42, 43, 44], params=dict(min_data_in_leaf=500)),
    "lr_l63_3s": dict(objective="lambdarank", seeds=[42, 43, 44], params=dict(num_leaves=63, min_data_in_leaf=100)),
    # targeted features (actionability) and run-length weighting, 3 seeds, both blend components
    "lgb_binary_act_3s": dict(objective="binary", seeds=[42, 43, 44], groups=["act"]),
    "lgb_lambdarank_act_3s": dict(objective="lambdarank", seeds=[42, 43, 44], groups=["act"]),
    "lgb_binary_typed_3s": dict(objective="binary", seeds=[42, 43, 44], groups=["typed"]),
    "lgb_lambdarank_typed_3s": dict(objective="lambdarank", seeds=[42, 43, 44], groups=["typed"]),
    "lgb_binary_rw5_3s": dict(objective="binary", seeds=[42, 43, 44], run_weight=0.5),
    "lgb_lambdarank_rw5_3s": dict(objective="lambdarank", seeds=[42, 43, 44], run_weight=0.5),
    "lgb_lambdarank_rw10_3s": dict(objective="lambdarank", seeds=[42, 43, 44], run_weight=1.0),
    "lgb_binary_rw10_3s": dict(objective="binary", seeds=[42, 43, 44], run_weight=1.0),
    # XGBoost on the GPU (CUDA), 3 seeds
    "xgb_pairwise_gpu_3s": dict(lib="xgb", objective="rank:pairwise", device="cuda", seeds=[42, 43, 44]),
    "xgb_ndcg_gpu_3s": dict(lib="xgb", objective="rank:ndcg", device="cuda", seeds=[42, 43, 44]),
    "xgb_pairwise_rw5_gpu_3s": dict(lib="xgb", objective="rank:pairwise", device="cuda", seeds=[42, 43, 44],
                                    run_weight=0.5),
    "xgb_binary_gpu_3s": dict(lib="xgb", objective="binary:logistic", device="cuda", seeds=[42, 43, 44]),
    # exit coordination (EXIT_COORDINATION_PROMPT.md): causal xc, forward-looking xc_lead (explicit opt-in future=True)
    "lgb_lambdarank_rw5_xc_3s": dict(objective="lambdarank", seeds=[42, 43, 44], run_weight=0.5, groups=["xc"]),
    "lgb_lambdarank_rw5_xcl_3s": dict(objective="lambdarank", seeds=[42, 43, 44], run_weight=0.5,
                                      groups=["xc", "xc_lead"], future=True),
    "lgb_binary_xcl_3s": dict(objective="binary", seeds=[42, 43, 44], groups=["xc", "xc_lead"], future=True),
    # block / task-lifetime features from later snapshots (explicit opt-in future=True)
    "lr_rw5_lead_3s": dict(objective="lambdarank", seeds=[42, 43, 44], run_weight=0.5, groups=["lead_blk"],
                           future=True),
    "bin_lead_3s": dict(objective="binary", seeds=[42, 43, 44], groups=["lead_blk"], future=True),
    "lr_rw5_leadchg_3s": dict(objective="lambdarank", seeds=[42, 43, 44], run_weight=0.5,
                              groups=["lead_blk", "lead_chg"], future=True),
    # DIAGNOSTIC ONLY (non-causal, violates ground rule 6): never refit/submitted without explicit approval
    "DIAG_lgb_binary_lead": dict(objective="binary", groups=["lead"], noncausal=True),
    # drop sample-level session-activity context (may encode controller style)
    "lgb_binary_noctx": dict(objective="binary", drop=["a_since_any", "a_n60", "a_n300"]),
    "lgb_lambdarank_noctx": dict(objective="lambdarank", drop=["a_since_any", "a_n60", "a_n300"]),
    # §8.D flagged experiment: out-of-fold (scenario, task_name) target encoding
    "lgb_binary_te": dict(objective="binary", te=20.0),
    "xgb_ndcg": dict(lib="xgb", objective="rank:ndcg"),
    "xgb_binary": dict(lib="xgb", objective="binary:logistic"),
    "cat_querysoftmax": dict(lib="cat", objective="QuerySoftMax", max_rounds=1000),
    "cat_yetirank": dict(lib="cat", objective="YetiRank", max_rounds=1000),
}


def _smooth_argmax(curves: list[list[float]], window: int = 25) -> tuple[int, np.ndarray]:
    n = min(len(c) for c in curves)
    mean = np.mean([c[:n] for c in curves], axis=0)
    sm = pd.Series(mean).rolling(window, min_periods=1, center=True).mean().to_numpy()
    return int(np.argmax(sm)) + 1, mean


def session_of(df: pd.DataFrame) -> pd.Series:
    return df.groupby("sample_id", sort=True)["session"].first()


def check_future(name: str, spec: dict) -> None:
    """Forward-looking feature groups (later snapshots of the session) require an explicit future=True opt-in."""
    fut = set(spec.get("groups", ())) & features.FUTURE_GROUPS
    if fut and not (spec.get("future") or spec.get("noncausal")):
        raise SystemExit(f"{name} uses forward-looking groups {sorted(fut)} without the opt-in future=True")


def run_cv(name: str, spec: dict, data: dict, cfg: dict, stage: str) -> dict:
    check_future(name, spec)
    tr, va = data["train"], data["validation"]
    groups = tuple(spec.get("groups", ()))
    drop = tuple(spec.get("drop", ()))
    trf, feats = features.build(tr, groups, drop)
    vaf, _ = features.build(va, groups, drop)
    folds = fold_array(tr, controller_folds(tr, cfg["n_folds"]))
    seeds = spec.get("seeds", [cfg["seed"]])
    max_rounds, es = spec.get("max_rounds", cfg["max_rounds"]), spec.get("es", cfg["early_stopping_rounds"])

    te_m = spec.get("te")
    te_feats = []
    t0 = time.time()
    curves, boosters, evals = [], [], {}
    for k in range(cfg["n_folds"]):
        trk, evk = trf[folds != k], trf[folds == k]
        if te_m:  # train rows: encoded from the other training controllers' folds; eval rows: from all training rows
            enc_tr = features.te_oof(trk, folds[folds != k], te_m)
            te_feats = list(enc_tr)
            evk = features.add_cols(evk, features.te_apply(evk, features.te_fit(trk, te_m)))
            trk = features.add_cols(trk, enc_tr)
        evals[k] = evk
        for s in seeds:
            b, c = models.fit(trk, feats + te_feats, spec, cfg["lgb_base"], max_rounds, seed=s,
                              n_jobs=cfg["n_jobs"], valid=evk, early_stopping=es)
            curves.append(c)
            boosters.append((k, s, b))
    R, mean_curve = _smooth_argmax(curves)
    oof = np.zeros(len(trf))
    for k, s, b in boosters:
        m = folds == k
        oof[m] += models.predict(b, evals[k], feats + te_feats, num_iteration=R) / len(seeds)
    cv_time = time.time() - t0
    fold_scores = [float(hits_from_scores(trf[folds == k], oof[folds == k]).mean()) for k in range(cfg["n_folds"])]

    # confirmation on validation: one fit on all of train, rounds scaled by data size
    n_fold_train = np.mean([(folds != k).sum() for k in range(cfg["n_folds"])])
    R_full = int(round(R * len(trf) / n_fold_train))
    if te_m:
        vaf = features.add_cols(vaf, features.te_apply(vaf, features.te_fit(trf, te_m)))
        trf = features.add_cols(trf, features.te_oof(trf, folds, te_m))
        feats = feats + te_feats
    t1 = time.time()
    val_score = np.zeros(len(vaf))
    for s in seeds:
        b, _ = models.fit(trf, feats, spec, cfg["lgb_base"], R_full, seed=s, n_jobs=cfg["n_jobs"])
        t2 = time.time()
        val_score += models.predict(b, vaf, feats) / len(seeds)
        infer = time.time() - t2
    train_time = time.time() - t1
    vh = hits_from_scores(vaf, val_score)
    v_mean, v_lo, v_hi = session_bootstrap(vh, session_of(vaf))
    oh = hits_from_scores(trf, oof)
    o_mean, o_lo, o_hi = session_bootstrap(oh, session_of(trf))

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(PRED_DIR / f"{name}.npz", oof=oof, val=val_score, R=R, R_full=R_full, feats=np.array(feats))
    if spec.get("lib", "lgb") == "lgb":
        imp = pd.Series(b.feature_importance("gain"), index=feats).sort_values(ascending=False)
        imp.to_csv(PRED_DIR / f"{name}_gain.csv")
    row = log_experiment(stage, name, dict(spec=spec, lgb_base=cfg["lgb_base"], n_feats=len(feats)), fold_scores,
                         float(vh.mean()), train_time, infer, notes=f"R={R} R_full={R_full}",
                         cv_time_s=round(cv_time, 1), oof_ci=f"[{o_lo:.4f},{o_hi:.4f}]",
                         val_ci=f"[{v_lo:.4f},{v_hi:.4f}]", R=R, R_full=R_full)
    print(f"[{name}] CV {row['cv_mean']:.4f} ± {row['cv_std']:.4f} folds {[round(x, 4) for x in fold_scores]} "
          f"| val {vh.mean():.4f} CI [{v_lo:.4f},{v_hi:.4f}] | R={R} R_full={R_full} | cv {cv_time:.0f}s "
          f"full-fit {train_time:.0f}s | curve max {mean_curve.max():.4f}")
    return row


def run_baselines(data: dict, cfg: dict) -> None:
    tr, va = data["train"], data["validation"]
    mapping = controller_folds(tr, cfg["n_folds"])
    folds = fold_array(tr, mapping)
    rep = fold_report(tr, mapping)
    rep.to_csv(ROOT / "reports" / "folds.csv", index=False)
    print(rep.to_string())
    vs = session_of(va)

    # random: average over seeds
    vals = [hits_from_scores(va, baselines.random_score(va, s)).mean() for s in range(20)]
    cvs = [[hits_from_scores(tr[folds == k], baselines.random_score(tr[folds == k], s)).mean() for k in range(4)]
           for s in range(5)]
    log_experiment("1", "random", dict(seeds=20), list(np.mean(cvs, axis=0)), float(np.mean(vals)),
                   notes=f"val std over 20 seeds {np.std(vals):.4f}; 3/mean seq_len val = "
                         f"{3 / va.groupby('sample_id').seq_len.first().mean():.4f}")
    print(f"random: val {np.mean(vals):.4f} (3/mean_seq {3 / va.groupby('sample_id').seq_len.first().mean():.4f})")

    for nm, fn in (("rule_baseline", baselines.rule_score), ("type_prior", baselines.type_prior_score)):
        fs = []
        for k in range(cfg["n_folds"]):
            t0 = time.time()
            fs.append(float(hits_from_scores(tr[folds == k], fn(tr[folds != k], tr[folds == k])).mean()))
        t0 = time.time()
        vsc = fn(tr, va)
        it = time.time() - t0
        vh = hits_from_scores(va, vsc)
        m, lo, hi = session_bootstrap(vh, vs)
        np.savez(PRED_DIR / f"{nm}.npz", val=vsc)
        row = log_experiment("1", nm, dict(rule=nm), fs, m, 0.0, it, val_ci=f"[{lo:.4f},{hi:.4f}]")
        print(f"{nm}: CV {row['cv_mean']:.4f} ± {row['cv_std']:.4f} {[round(x, 4) for x in fs]} | val {m:.4f} "
              f"[{lo:.4f},{hi:.4f}]")


def run_final(name: str, spec: dict, data: dict, cfg: dict) -> None:
    """Refit on train + validation (the documented, permitted use) with rounds fixed from CV."""
    if spec.get("noncausal"):  # diagnostic-only experiments stay blocked
        raise SystemExit(f"{name} is a diagnostic-only experiment; refusing to build a final model")
    check_future(name, spec)  # forward-looking groups are allowed only with the explicit opt-in future=True
    import lightgbm as lgb  # noqa: F401
    pred = np.load(PRED_DIR / f"{name}.npz")
    R_full = int(pred["R_full"])
    full = pd.concat([data["train"], data["validation"]], ignore_index=True)
    full = full.sort_values(["sample_id", "row_index"], kind="stable").reset_index(drop=True)
    groups = tuple(spec.get("groups", ()))
    fullf, feats = features.build(full, groups, tuple(spec.get("drop", ())))
    R_final = int(round(R_full * len(fullf) / len(data["train"])))
    te_m = spec.get("te")
    if te_m:  # validation controllers form one extra group for the out-of-group encoding
        tr_folds = controller_folds(data["train"], cfg["n_folds"])
        grp = fullf["atco"].map(tr_folds).fillna(cfg["n_folds"]).to_numpy().astype(int)
        enc_cols = features.te_oof(fullf, grp, te_m)
        fullf = features.add_cols(fullf, enc_cols)
        feats = feats + list(enc_cols)
    seeds = spec.get("final_seeds", [cfg["seed"], cfg["seed"] + 1, cfg["seed"] + 2])
    out = ROOT / cfg["paths"]["artifacts_dir"] / name
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for s in seeds:
        b, _ = models.fit(fullf, feats, spec, cfg["lgb_base"], R_final, seed=s, n_jobs=cfg["n_jobs"])
        models.save(b, str(out / f"model_seed{s}"), spec.get("lib", "lgb"))
    if te_m:
        import pickle
        (out / "te_encoder.pkl").write_bytes(pickle.dumps(features.te_fit(full, te_m)))
    meta = dict(name=name, spec=spec, feats=feats, groups=list(groups), R_final=R_final, seeds=seeds,
                lgb_base=cfg["lgb_base"], trained_on="train+validation", train_time_s=round(time.time() - t0, 1))
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    log_experiment("8", f"final:{name}", meta, notes=f"R_final={R_final} seeds={seeds}",
                   train_time=time.time() - t0)
    print(f"final {name}: R_final={R_final} seeds={seeds} saved to {out}")


def main(argv: list[str]) -> None:
    cfg = load_config()
    np.random.seed(cfg["seed"])
    data = load_all(cfg)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    cmd = argv[0]
    if cmd == "baselines":
        run_baselines(data, cfg)
    elif cmd == "cv":
        stage = "cv"
        for name in argv[1:]:
            run_cv(name, EXPERIMENTS[name], data, cfg, stage)
    elif cmd == "final":
        run_final(argv[1], EXPERIMENTS[argv[1]], data, cfg)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
