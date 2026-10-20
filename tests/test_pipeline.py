import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from src.baseline import baseline_score
from src.config import DROP_COLS, ROOT_DIR
from src.cv import make_folds
from src.data import load, load_trainval
from src.features import VERSIONS, build_features, parse_callsigns
from src.metric import to_submission, top3_hit_rate
from src.models.lgbm_rank import group_sizes, hit3_from_arrays

LATEST = list(VERSIONS)[-1]


def test_resolve_families():
    from src.features import resolve
    assert resolve("v2") == ["tts", "snap"]
    assert resolve("ac+load") == ["tts", "load", "ac"]
    with pytest.raises(ValueError):
        resolve("foo")


@pytest.fixture(scope="module")
def val():
    return load("validation")


def test_metric_matches_scorer(val, tmp_path):
    val = val.assign(score=baseline_score(val))
    sub = tmp_path / "sub.csv"
    to_submission(val, "score").to_csv(sub, index=False)
    out = subprocess.run([sys.executable, str(ROOT_DIR / "scorer.py"), "--labelled",
                          str(ROOT_DIR / "data" / "validation.csv"), str(sub)], capture_output=True, text=True)
    official = float(out.stdout.split("top-3 hit rate:")[1].split()[0])
    assert round(top3_hit_rate(val, "score"), 4) == official


def test_one_target_per_sample(val):
    assert (val.groupby("sample_id")["target_index"].sum() == 1).all()


def test_group_sizes_aligned(val):
    g = group_sizes(val["sample_id"])
    assert g.sum() == len(val)
    assert (g == val.groupby("sample_id", sort=False).size().to_numpy()).all()
    with pytest.raises(ValueError):
        group_sizes(val["sample_id"].sample(frac=1, random_state=0))


def test_hit3_arrays_matches_metric(val):
    s = np.random.default_rng(0).normal(size=len(val))
    fast = hit3_from_arrays(s, val["target_index"].to_numpy(), group_sizes(val["sample_id"]))
    assert abs(fast - top3_hit_rate(val.assign(s=s), "s")) < 1e-9


def test_folds_split_by_atco():
    df = load_trainval()
    fold = make_folds(df)
    assert (fold >= 0).all()
    assert (df.groupby("atco")["atco"].transform("size") > 0).all()
    assert df.assign(f=fold).groupby("atco")["f"].nunique().max() == 1
    assert df.assign(f=fold).groupby("session")["f"].nunique().max() == 1


def test_no_forbidden_features(val):
    _, feats = build_features(val, LATEST)
    assert not set(feats) & set(DROP_COLS)
    assert len(feats) == len(set(feats))


def test_features_do_not_depend_on_target(val):
    a, feats = build_features(val, LATEST)
    shuffled = val.assign(target_index=val.groupby("sample_id")["target_index"].transform(np.random.permutation))
    b, _ = build_features(shuffled, LATEST)
    pd.testing.assert_frame_equal(a[feats], b[feats])


def test_features_preserve_rows_and_order(val):
    out, _ = build_features(val, LATEST)
    assert len(out) == len(val)
    assert (out["sample_id"].to_numpy() == val["sample_id"].to_numpy()).all()
    assert (out["row_index"].to_numpy() == val["row_index"].to_numpy()).all()


def test_parse_callsigns():
    names = pd.Series(["assume_aircraft_SAS77A", "conflict_resolution_QTR50U_GFM7Z", "entry_conflict_resolution_A1_B2"])
    cs = parse_callsigns(names, pd.Series([4, 6, 0]))
    assert cs["callsign"].tolist() == ["SAS77A", "QTR50U", "A1"]
    assert cs["callsign2"].tolist()[1:] == ["GFM7Z", "B2"]


def test_check_submission_accepts_official_sample():
    from src.make_submission import check_submission
    sub = pd.read_csv(ROOT_DIR / "data" / "sample_submission.csv", dtype={"prediction": str})
    check_submission(sub, load("test"))
    with pytest.raises(AssertionError):
        check_submission(sub.iloc[1:], load("test"))
