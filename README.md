# AWARE active-task prediction — reproducible pipeline

Ranks each candidate task within a snapshot and writes the 3 best `row_index` values per test sample.
Model selection uses leave-controller-out CV on `train.csv` only; `validation.csv` is a one-shot
confirmation holdout and is merged into training only for the final refit. See `reports/REPORT.md`.

## Data

The competition files are not tracked in git. Place `train.csv`, `validation.csv`, `test.csv` and
`sample_submission.csv` in the repository root before running the pipeline. Generated folders
(`cache/`, `artifacts/`, `logs/`, `submissions/`) are also ignored and are recreated by the commands below.

## Environment

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt      # Python 3.12.3; pinned versions
```

All commands run from the repository root as modules (`python -m src.<module>`). `src/data_io.py`
is deliberately not called `io.py`, so it can't shadow the standard library.

## End-to-end reproduction

```bash
PY=.venv/bin/python
$PY -m src.validate_data                         # Stage 0 audit  -> reports/data_audit.md (+ Parquet cache)
$PY -m tests.test_metric_vs_scorer               # internal metric == scorer.py
$PY -m src.train baselines                       # Stage 1: random / rule (reproduces sample_submission) / type prior
$PY -m src.train cv lgb_binary                   # Stage 2: reference model (CV + validation confirmation)
$PY -m src.train cv lgb_lambdarank lgb_xendcg lgb_softmax lgb_binary_w     # Stage 4 objectives
$PY -m src.train cv lgb_binary_rel_ac lgb_binary_rel lgb_binary_ac lgb_binary_temp   # Stage 5/6 features
$PY -m src.tune_smoothing lgb_binary             # Stage 6 causal score smoothing (CV-tuned)
$PY -m src.compare <candidate> <reference>       # §6.5 adoption check -> reports/comparisons.csv
$PY -m src.ensemble <exp_a> <exp_b> --method z   # Stage 7 blends (weights from CV OOF only)
$PY -m src.error_analysis <experiment>           # -> reports/error_analysis.md
$PY -m src.predict <experiment> --val-from-cv    # validation submission from the train-only model
$PY scorer.py --labelled validation.csv submissions/validation_submission.csv
$PY -m src.train final <experiment>              # Stage 8 refit on train+validation -> artifacts/<experiment>/
$PY -m src.predict <experiment>                  # -> submissions/submission.csv (validated after writing)
$PY -m src.validate_submission submissions/submission.csv
```

**Final model:** equal-weight within-sample z-score blend of `lgb_binary_3s` and `lgb_lambdarank_3s`
(3 seeds each, refit on train+validation). Exact commands: end of `reports/REPORT.md`.

## Layout

```
config/default.yaml      paths, seed, folds, LightGBM base params
src/data_io.py           typed loading, Parquet cache, group checks
src/validate_data.py     Stage 0 audit
src/metrics.py           top-3 hit rate (== scorer.py), session-block bootstrap
src/cv.py                leave-controller-out folds (GroupKFold on atco)
src/baselines.py         random, rule (exact sample_submission reproduction), type prior
src/features.py          base + within-sample (rel) + aircraft (ac) + causal temporal (temp) features
src/models.py            LightGBM binary/lambdarank/xendcg/softmax, XGBoost, CatBoost wrappers, top-3 feval
src/rank.py              within-sample ranking, top-3 extraction, submission writing
src/temporal.py          causal score smoothing
src/tune_smoothing.py    CV tuning of smoothing
src/ensemble.py          within-sample z/rank blending (CV-weighted)
src/blend_fixed.py       pre-specified equal-weight blend (the selected final model)
src/importance.py        per-fold gain + held-out permutation importance
src/compare.py           paired fold / session-bootstrap comparison
src/error_analysis.py    §12 slices
tests/test_metric_vs_scorer.py  internal metric == scorer.py
src/train.py             CV experiments + final refit, logs to reports/experiments.csv
src/predict.py           artifacts -> submission (no training code)
src/validate_submission.py  strict submission validator
```

Seeds: global seed 42 (config); final models average seeds 42, 43, 44. LightGBM runs with `deterministic=true`.
