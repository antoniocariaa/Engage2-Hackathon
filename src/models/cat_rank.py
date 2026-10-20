"""CatBoost QuerySoftMax: softmax sulle righe dello snapshot, task_index categorico nativo."""
import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool

from src.config import N_THREADS, RANDOM_SEED, USE_GPU
from src.features import CAT_FEATURES
from src.models.lgbm_rank import group_sizes

PARAMS = {
    "loss_function": "QuerySoftMax",
    # leggero per CPU: con 800 iterazioni la CV a 5 fold richiederebbe >2 h su 8 core
    "iterations": 300,
    "learning_rate": 0.15,
    "depth": 6,
    "border_count": 64,
    "l2_leaf_reg": 3.0,
    "thread_count": N_THREADS,
    "random_seed": RANDOM_SEED,
    "verbose": 0,
}
if USE_GPU:
    PARAMS["task_type"] = "GPU"


def _pool(X, y, sid, weight=None):
    X = X.copy()
    cats = [c for c in CAT_FEATURES if c in X.columns]
    for c in cats:
        X[c] = X[c].astype(int)
    group_sizes(sid)  # verifica contiguità
    return Pool(X, label=y, group_id=sid.to_numpy(), cat_features=cats, group_weight=weight)


class CatRank:
    name = "cat_rank"

    def __init__(self, params=None):
        self.params = {**PARAMS, **(params or {})}
        self.model = None
        self.curve = None

    def fit(self, X, y, sid, X_val=None, y_val=None, sid_val=None, num_boost_round=None, weight=None):
        # num_boost_round ignorato: CatBoost usa "iterations" (round fissi, niente early stopping sul fold di valutazione)
        self.model = CatBoostRanker(**self.params)
        self.model.fit(_pool(X, y, sid, weight))
        return self

    def predict(self, X):
        X = X.copy()
        for c in [c for c in CAT_FEATURES if c in X.columns]:
            X[c] = X[c].astype(int)
        return self.model.predict(X)

    def importance(self, feats) -> pd.Series:
        # il default per i ranker (LossFunctionChange) richiede il dataset di training
        imp = self.model.get_feature_importance(type="PredictionValuesChange")
        return pd.Series(imp, index=feats).sort_values(ascending=False)
