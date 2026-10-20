"""LightGBM LambdaRank con early stopping sul top-3 hit rate."""
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import RANDOM_SEED
from src.features import CAT_FEATURES

PARAMS = {
    "objective": "lambdarank",
    "metric": "None",
    "lambdarank_truncation_level": 10,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "num_threads": 8,
    "seed": RANDOM_SEED,
    "verbose": -1,
}
# Numero di round fisso: niente early stopping sul fold di valutazione (renderebbe l'OOF ottimista).
# Il fold di valutazione serve solo a registrare la curva hit3, usata per scegliere NUM_BOOST_ROUND sulla CV media.
NUM_BOOST_ROUND = 300


def group_sizes(sample_id: pd.Series) -> np.ndarray:
    """Dimensioni dei gruppi nell'ordine delle righe; fallisce se i sample_id non sono contigui."""
    sid = sample_id.to_numpy()
    starts = np.r_[True, sid[1:] != sid[:-1]]
    if starts.sum() != len(np.unique(sid)):
        raise ValueError("sample_id non contigui: ordinare per sample_id prima di costruire i gruppi")
    return np.diff(np.r_[np.flatnonzero(starts), len(sid)])


def hit3_from_arrays(scores: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> float:
    """Top-3 hit rate vettoriale: il target è nel top-3 se meno di 3 righe del gruppo hanno score maggiore."""
    gid = np.repeat(np.arange(len(groups)), groups)
    tgt = np.zeros(len(groups))
    tgt[gid[labels == 1]] = scores[labels == 1]
    better = np.bincount(gid, weights=(scores > tgt[gid]).astype(float), minlength=len(groups))
    return float(np.mean(better < 3))


def _feval(groups):
    def f(preds, data):
        return "hit3", hit3_from_arrays(preds, data.get_label(), groups), True
    return f


class LGBMRank:
    name = "lgbm_rank"

    def __init__(self, params=None):
        self.params = {**PARAMS, **(params or {})}
        self.model = None
        self.curve = None

    def fit(self, X, y, sid, X_val=None, y_val=None, sid_val=None, num_boost_round=NUM_BOOST_ROUND, weight=None):
        cats = [c for c in CAT_FEATURES if c in X.columns]
        dtr = lgb.Dataset(X, y, group=group_sizes(sid), weight=weight, categorical_feature=cats, free_raw_data=False)
        kw, rec = {}, {}
        if X_val is not None:
            gv = group_sizes(sid_val)
            dva = lgb.Dataset(X_val, y_val, group=gv, reference=dtr, categorical_feature=cats)
            kw = {"valid_sets": [dva], "valid_names": ["val"], "feval": _feval(gv),
                  "callbacks": [lgb.record_evaluation(rec)]}
        self.model = lgb.train(self.params, dtr, num_boost_round=num_boost_round, **kw)
        self.curve = np.array(rec["val"]["hit3"]) if rec else None
        return self

    def predict(self, X):
        return self.model.predict(X)

    def importance(self, feats) -> pd.Series:
        return pd.Series(self.model.feature_importance("gain"), index=feats).sort_values(ascending=False)


class LGBMXendcg(LGBMRank):
    """Variante listwise (cross-entropy NDCG): più vicina alla softmax sullo snapshot."""
    name = "lgbm_xendcg"

    def __init__(self, params=None):
        super().__init__({"objective": "rank_xendcg", **(params or {})})
