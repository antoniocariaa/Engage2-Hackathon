"""Submission finale (M8): riaddestra i modelli dell'ensemble su train+validation e predice test.csv.

    python -m src.make_submission --ensemble experiments/ensemble_ens-1.json --out experiments/sub/ens-1.csv
    python -m src.make_submission --oof abl-ac --out experiments/sub/lgbm-ac.csv      (singolo modello)
"""
import argparse
import json
import subprocess
import sys

import numpy as np
import pandas as pd

from src.config import DATA_DIR, OOF_DIR, ROOT_DIR, SUB_DIR
from src.cv import MODELS
from src.data import load, load_trainval
from src.ensemble import load_oofs, normalize
from src.episodes import sample_weights
from src.features import build_features
from src.metric import to_submission

N_TEST_SAMPLES = 1863


def check_submission(sub: pd.DataFrame, test: pd.DataFrame) -> None:
    """Fallisce se il file non è valido per Kaggle."""
    assert list(sub.columns) == ["sample_id", "prediction"], sub.columns
    assert sub["sample_id"].is_unique, "sample_id duplicati"
    assert set(sub["sample_id"]) == set(test["sample_id"]), "sample_id mancanti o in più"
    assert len(sub) == N_TEST_SAMPLES, len(sub)
    rows = test.groupby("sample_id")["row_index"].apply(set)
    for sid, pred in zip(sub["sample_id"], sub["prediction"]):
        ids = [int(x) for x in pred.split()]
        assert len(set(ids)) == 3, (sid, pred)
        assert set(ids) <= rows[sid], (sid, pred)


def fit_predict(meta: dict, trainval: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    tr, feats = build_features(trainval, meta["features"])
    te, _ = build_features(test, meta["features"])
    te["task_index"] = te["task_index"].cat.set_categories(tr["task_index"].cat.categories)
    w = sample_weights(tr, meta["alpha"]) if meta["alpha"] > 0 else None
    m = MODELS[meta["model"]]().fit(tr[feats], tr["target_index"], tr["sample_id"],
                                    num_boost_round=meta["rounds"], weight=w)
    return m.predict(te[feats])


def validation_sanity(names, weights, out_path):
    """Lo stesso blend sugli OOF delle righe di validation.csv, passato a scorer.py: verifica il formato end-to-end."""
    oof = load_oofs(names)
    oof["s"] = oof[names].to_numpy() @ np.asarray(weights)
    val_ids = set(load("validation")["sample_id"])
    sub = to_submission(oof[oof["sample_id"].isin(val_ids)], "s")
    path = out_path.with_name(out_path.stem + "_val_oof.csv")
    sub.to_csv(path, index=False)
    subprocess.run([sys.executable, str(ROOT_DIR / "scorer.py"), "--labelled", str(DATA_DIR / "validation.csv"),
                    str(path)], check=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ensemble", help="json scritto da src.ensemble")
    g.add_argument("--oof", help="un solo esperimento (exp_id)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.ensemble:
        ens = json.load(open(args.ensemble))
        names, weights = ens["models"], ens["weights"]
    else:
        names, weights = [args.oof], [1.0]
    out = ROOT_DIR / args.out
    trainval, test = load_trainval(), load("test")
    test = test.sort_values(["sample_id", "row_index"]).reset_index(drop=True)
    blend = np.zeros(len(test))
    for n, w in zip(names, weights):
        if w == 0:
            continue
        cache = SUB_DIR / f"test_pred_{n}.parquet"
        if cache.exists():
            # modello già addestrato su train+validation: riuso le predizioni (max 2 addestramenti per modello)
            pred = test[["sample_id", "row_index"]].merge(pd.read_parquet(cache), on=["sample_id", "row_index"])
            print(f"cache {n}: {cache.name}", flush=True)
            z = pred["z"].to_numpy()
        else:
            meta = json.load(open(OOF_DIR / f"{n}.json"))
            print(f"fit {n}: {meta}", flush=True)
            test["_s"] = fit_predict(meta, trainval, test)
            z = normalize(test, "_s").to_numpy()
            test[["sample_id", "row_index"]].assign(z=z).to_parquet(cache, index=False)
        blend += w * z
    test["score"] = blend
    sub = to_submission(test, "score")
    check_submission(sub, test)
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False)
    test[["sample_id", "row_index", "score"]].to_parquet(out.with_suffix(".scores.parquet"), index=False)
    print(f"ok: {out} ({len(sub)} righe, check superato)")
    validation_sanity(names, weights, out)
