"""LightGBM wrappers (binary / lambdarank / rank_xendcg / per-sample softmax) with a top-3 feval."""
from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features import CAT_FEATURES

RANK_OBJECTIVES = {"lambdarank", "rank_xendcg"}


@dataclass
class Groups:
    """Group structure of a frame sorted by sample_id."""
    starts: np.ndarray
    sizes: np.ndarray
    row_index: np.ndarray
    tpos: np.ndarray | None  # absolute position of the labelled row per group

    @classmethod
    def of(cls, df: pd.DataFrame) -> "Groups":
        sid = df["sample_id"].to_numpy()
        if np.any(np.diff(sid) < 0):
            raise ValueError("frame must be sorted by sample_id")
        first = np.r_[True, sid[1:] != sid[:-1]]
        starts = np.flatnonzero(first)
        sizes = np.diff(np.r_[starts, len(sid)])
        if not np.array_equal(sizes, df["seq_len"].to_numpy()[starts]):
            raise ValueError("group sizes do not match seq_len")
        tpos = None
        if "target_index" in df:
            tpos = np.flatnonzero(df["target_index"].to_numpy() == 1)
            if len(tpos) != len(starts):
                raise ValueError("expected exactly one positive per sample")
        return cls(starts, sizes, df["row_index"].to_numpy(), tpos)


def top3_hits(score: np.ndarray, g: Groups, k: int = 3) -> np.ndarray:
    """Per-sample hit with the same tie-break as rank.topk (higher score, then lower row_index)."""
    s_t = np.repeat(score[g.tpos], g.sizes)
    r_t = np.repeat(g.row_index[g.tpos], g.sizes)
    beats = (score > s_t) | ((score == s_t) & (g.row_index < r_t))
    return np.add.reduceat(beats.astype(np.int32), g.starts) < k


def make_feval(groups_by_name: dict[str, Groups]):
    def feval(preds, data):
        g = groups_by_name[data._aware_name]
        return "top3", float(top3_hits(np.asarray(preds, dtype=np.float64), g).mean()), True
    return feval


def softmax_objective(g: Groups):
    """Per-sample softmax cross-entropy: one positive per group, each sample weighted equally."""
    def obj(preds, data):
        y = data.get_label()
        m = np.repeat(np.maximum.reduceat(preds, g.starts), g.sizes)
        e = np.exp(preds - m)
        p = e / np.repeat(np.add.reduceat(e, g.starts), g.sizes)
        grad = p - y
        hess = np.maximum(p * (1.0 - p), 1e-6)
        return grad, hess
    return obj


def _dataset(df: pd.DataFrame, feats: list[str], spec: dict, name: str, reference=None) -> lgb.Dataset:
    X = df[feats].to_numpy(dtype=np.float32)
    y = df["target_index"].to_numpy(dtype=np.float32) if "target_index" in df else None
    weight = None
    if spec.get("neg_weight") and y is not None:
        n = df["seq_len"].to_numpy(dtype=np.float64)
        w = {"inv_seq": 1.0 / n, "inv_seq_m1": 1.0 / (n - 1)}[spec["neg_weight"]]
        weight = np.where(y == 1, 1.0, w * spec.get("neg_weight_scale", 1.0)).astype(np.float32)
    if spec.get("run_weight") and y is not None:
        rw = run_length_weight(df, spec["run_weight"])
        weight = rw if weight is None else weight * rw
    group = Groups.of(df).sizes if spec["objective"] in RANK_OBJECTIVES or spec["objective"] == "softmax" else None
    cat = [feats.index(c) for c in CAT_FEATURES if c in feats]
    ds = lgb.Dataset(X, label=y, weight=weight, group=group, feature_name=feats, categorical_feature=cat,
                     reference=reference, free_raw_data=False)
    ds._aware_name = name
    return ds


def run_length_weight(df: pd.DataFrame, power: float) -> np.ndarray:
    """Per-row weight 1/run_len**power, where run_len is the number of consecutive samples of the same session
    sharing the same target task_name (near-duplicate snapshots count less). Mean weight rescaled to 1."""
    t = df.loc[df["target_index"] == 1, ["sample_id", "session", "task_name"]].sort_values("sample_id")
    key = t["session"].astype(str) + "|" + t["task_name"].astype(str)
    run = (key != key.shift()).cumsum()
    run_len = run.map(run.value_counts()).to_numpy(dtype=np.float64)
    w = pd.Series(run_len ** -power, index=t["sample_id"].to_numpy())
    w = w / w.mean()
    return df["sample_id"].map(w).to_numpy(dtype=np.float32)


def lgb_params(spec: dict, base: dict, seed: int, n_jobs: int) -> dict:
    p = dict(base)
    p.update(spec.get("params", {}))
    p.update(seed=seed, num_threads=n_jobs, metric="None")
    obj = spec["objective"]
    if obj != "softmax":
        p["objective"] = obj
    if obj == "lambdarank":
        p.setdefault("lambdarank_truncation_level", 10)
    return p


def fit(train: pd.DataFrame, feats: list[str], spec: dict, base: dict, num_rounds: int, seed: int = 42,
        n_jobs: int = 8, valid: pd.DataFrame | None = None, early_stopping: int | None = None):
    """Train one model. With `valid`, records the top-3 curve per iteration (returned as list)."""
    lib = spec.get("lib", "lgb")
    if lib == "xgb":
        return fit_xgb(train, feats, spec, num_rounds, seed, n_jobs, valid, early_stopping)
    if lib == "cat":
        return fit_cat(train, feats, spec, num_rounds, seed, n_jobs, valid)
    params = lgb_params(spec, base, seed, n_jobs)
    dtr = _dataset(train, feats, spec, "train")
    if spec["objective"] == "softmax":
        params["objective"] = softmax_objective(Groups.of(train))
    callbacks, curve = [], {}
    valid_sets, names, feval = [], [], None
    if valid is not None:
        dva = _dataset(valid, feats, spec, "valid", reference=dtr)
        valid_sets, names = [dva], ["valid"]
        feval = make_feval({"valid": Groups.of(valid)})
        callbacks.append(lgb.record_evaluation(curve))
        if early_stopping:
            callbacks.append(lgb.early_stopping(early_stopping, first_metric_only=True, verbose=False))
    booster = lgb.train(params, dtr, num_boost_round=num_rounds, valid_sets=valid_sets, valid_names=names,
                        feval=feval, callbacks=callbacks)
    return booster, curve.get("valid", {}).get("top3", [])


def predict(booster, df: pd.DataFrame, feats: list[str], num_iteration: int | None = None) -> np.ndarray:
    if isinstance(booster, lgb.Booster):
        if list(booster.feature_name()) != list(feats):
            raise ValueError("feature schema mismatch between model and data")
        return booster.predict(df[feats].to_numpy(dtype=np.float32), num_iteration=num_iteration, raw_score=True)
    import xgboost as xgb
    if isinstance(booster, xgb.Booster):
        if list(booster.feature_names) != list(feats):
            raise ValueError("feature schema mismatch between model and data")
        it = (0, num_iteration) if num_iteration else (0, 0)
        return booster.predict(_dmatrix(df, feats), iteration_range=it, output_margin=True)
    if list(booster.feature_names_) != list(feats):
        raise ValueError("feature schema mismatch between model and data")
    ntree_end = num_iteration or 0
    return booster.predict(_cat_pool(df, feats), prediction_type="RawFormulaVal", ntree_end=ntree_end)


# ---------------------------------------------------------------- XGBoost
def _dmatrix(df: pd.DataFrame, feats: list[str], with_label: bool = False, ranking: bool = False,
             run_weight: float | None = None):
    import xgboost as xgb
    X = df[feats].to_numpy(dtype=np.float32)
    ft = ["c" if f in CAT_FEATURES else "q" for f in feats]
    kw = {}
    if run_weight:
        w = run_length_weight(df, run_weight)
        kw["weight"] = w[Groups.of(df).starts] if ranking else w  # XGBoost ranking takes one weight per group
    if with_label:
        kw["label"] = df["target_index"].to_numpy(dtype=np.float32)
    if ranking:
        kw["qid"] = df["sample_id"].to_numpy()
    return xgb.DMatrix(X, feature_names=feats, feature_types=ft, enable_categorical=True, nthread=-1, **kw)


def fit_xgb(train, feats, spec, num_rounds, seed, n_jobs, valid, early_stopping):
    import xgboost as xgb
    ranking = spec["objective"].startswith("rank:")
    params = dict(objective=spec["objective"], eta=0.05, max_depth=6, min_child_weight=20, subsample=0.8,
                  colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist", max_cat_to_onehot=1, seed=seed,
                  nthread=n_jobs, disable_default_eval_metric=1, device=spec.get("device", "cpu"))
    if ranking:
        params.update(lambdarank_pair_method="topk", lambdarank_num_pair_per_sample=10)
    params.update(spec.get("params", {}))
    dtr = _dmatrix(train, feats, True, ranking, spec.get("run_weight"))
    evals, res, kw = [], {}, {}
    if valid is not None:
        g = Groups.of(valid)
        dva = _dmatrix(valid, feats, True, ranking)
        evals = [(dva, "valid")]
        kw["custom_metric"] = lambda preds, d: ("top3", float(top3_hits(np.asarray(preds, np.float64), g).mean()))
        kw["maximize"] = True
        if early_stopping:
            kw["early_stopping_rounds"] = early_stopping
    b = xgb.train(params, dtr, num_boost_round=num_rounds, evals=evals, evals_result=res, verbose_eval=False, **kw)
    return b, res.get("valid", {}).get("top3", [])


# ---------------------------------------------------------------- CatBoost
def _cat_pool(df: pd.DataFrame, feats: list[str], with_label: bool = False):
    from catboost import Pool
    X = df[feats].copy()
    for c in CAT_FEATURES:
        if c in X:
            X[c] = X[c].astype(int)
    y = df["target_index"].to_numpy() if with_label else None
    return Pool(X, label=y, group_id=df["sample_id"].to_numpy(), cat_features=[c for c in CAT_FEATURES if c in feats])


def fit_cat(train, feats, spec, num_rounds, seed, n_jobs, valid):
    from catboost import CatBoost
    params = dict(loss_function=spec["objective"], iterations=num_rounds, learning_rate=0.08, depth=6,
                  l2_leaf_reg=3.0, random_seed=seed, thread_count=n_jobs, verbose=False, allow_writing_files=False)
    params.update(spec.get("params", {}))
    m = CatBoost(params)
    m.fit(_cat_pool(train, feats, True))
    curve = []
    if valid is not None:
        pool, g = _cat_pool(valid, feats), Groups.of(valid)
        step = spec.get("eval_period", 25)
        prev = 0.0
        for p in m.staged_predict(pool, prediction_type="RawFormulaVal", eval_period=step):
            val = float(top3_hits(np.asarray(p, np.float64), g).mean())
            curve += [prev] * (step - 1) + [val]  # value sits at its exact iteration
            prev = val
    return m, curve


MODEL_EXT = {"lgb": "txt", "xgb": "json", "cat": "cbm"}


def save(model, path_stem: str, lib: str) -> None:
    model.save_model(f"{path_stem}.{MODEL_EXT[lib]}")


def load(path_stem: str, lib: str):
    path = f"{path_stem}.{MODEL_EXT[lib]}"
    if lib == "lgb":
        return lgb.Booster(model_file=path)
    if lib == "xgb":
        import xgboost as xgb
        b = xgb.Booster()
        b.load_model(path)
        return b
    from catboost import CatBoost
    return CatBoost().load_model(path)
