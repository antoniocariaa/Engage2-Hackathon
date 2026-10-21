# Implementation Prompt — AWARE Controller Active-Task Prediction

## 0. Role, objective, and ground rules

You are a senior machine-learning engineer. Build a reproducible, leakage-safe, efficient pipeline for the AWARE task-prediction challenge. The pipeline scores every candidate task in each sample, ranks the candidates within the sample, and writes the three highest-ranked distinct `row_index` values per test sample. The goal is the best **top-3 hit rate on unseen controllers**. The private leaderboard is controller 023, who never appears in any labelled file.

Ground rules (non-negotiable):

1. **Inspect before assuming.** Before writing modelling code, read `aware_task_prediction_organized.md` (the challenge documentation; its §8 is the column dictionary), `scorer.py`, and the actual CSV headers and contents. The docs refer to a `FEATURES.md` that was **not present** in the workspace when this prompt was written. If you find it, read it and report any differences from §8 of the Markdown file.
2. **No fabrication.** Every number you report must come from code you executed in this session, and it must be logged to a file. Never claim an experiment improved performance unless you ran it and measured it under the protocol in §6.
3. **No invented columns.** Use only columns that exist in the files. Any engineered feature must be computable at inference time from test features alone.
4. **Do not assume packages are installed.** When this prompt was written, the system `python3` lacked `pandas`, `lightgbm`, `xgboost`, `catboost`, and `scikit-learn`. Check for an existing venv or conda environment first. If installing is permitted, create a project venv and pin versions in `requirements.txt`. If installing is not possible, say so explicitly and deliver the best possible solution with what is available.
5. **No label leakage.** Validation and test labels must never be used for training, feature construction, feature selection, hyper-parameter tuning, early stopping, ensemble weighting, or post-processing tuning. The single exception is the final refit described in §7, Stage 8. Do not probe the leaderboard.
6. **Causal features only.** A sample's features may use that sample's rows and **earlier** snapshots of the same session. They must never use later snapshots, even though later test rows are physically present in `test.csv`.
7. **Time-box.** The competition page showed a closing deadline. Work in the stage order in §7. A valid submission that beats the baseline must exist as early as possible (by the end of Stage 3), and each later stage only replaces it if it is validated as better.
8. If a required file or capability is missing, state the limitation and continue with the most useful implementation possible.

## 1. Challenge summary

An en-route air traffic controller in Swedish upper-airspace sector `ESMM_23` works through a list of open tasks. AWARE generates that list automatically every 3 seconds. Experts annotated which task the controller was working on at each labelled moment.

- A **sample** (`sample_id`) is one task-list snapshot.
- A **row** is one candidate task in that list. `row_index` is the task's position, `0..seq_len-1`. `seq_len` ranges from 9 to 59, with a mean of about 36.
- Exactly one row per labelled sample has `target_index = 1`.
- Each task has a type (`task_index`, 14 codes, categorical) and an aircraft. `task_name` = `<task_type>_<callsign>`. Conflict tasks name two callsigns, and the first is the subject aircraft.
- Task: for each test sample, output three distinct `row_index` values, ordered best first.

## 2. Dataset facts

Documented, and confirmed by a preliminary inspection. **Re-verify all of them in Stage 0.**

| File | Sessions | Samples | Rows | Columns |
|---|---|---:|---:|---:|
| `train.csv` | 34 (16 controllers: 001–011, 014–018) | 11,923 | 432,578 | 96 |
| `validation.csv` | `019sc1`, `019sc2`, `020sc1` | 1,252 | 42,110 | 96 |
| `test.csv` | `020sc2`, `020sc3` (public, 915 samples), `023sc1`, `023sc2`, `023sc3` (private, 948 samples) | 1,863 | 66,409 | 95 (no `target_index`) |
| `sample_submission.csv` | — | 1,863 | — | `sample_id,prediction` |

- **Columns.** The 10 identifier/label columns are `sample_id, seq_len, row_index, target_index, task_index, session, atco, scenario, timestamp, task_name`. The remaining 86 are features in groups: aircraft position/status, conflicts, route/conformance, military zone, `x_` hand-over (8), `k_` aircraft state/exit (32), `a_` controller inputs (12), and `l_` list context (9). Column order differs from the documentation's table order, so always select columns by name.
- **Preliminary checks** (train and validation): rows per sample always equal `seq_len`, and `row_index` is contiguous from 0. `task_name` is unique within each sample. Every labelled sample has exactly one positive. Task type 11 never occurs.
- **Target type counts in train** (`task_index: positives / rows`): 5 transfer 3145/27,786; 4 assume 3003/71,633; 12 zone_conflict 1950/48,719; 2 entry_coordination 940/128,283; 13 return_to_route 881/15,277; 3 exit_coordination 704/54,845; 10 exit_conditions 407/33,400; 6 conflict_resolution 351/10,000; 8 qos 237/27,461; 1 exit_conflict 186/5,558; 7 non_conformance 86/1,297; 0 entry_conflict 31/8,194; 9 entry_conditions 2/125.
- **Twin pair.** About 30% of train targets (3,535 of 11,923) belong to an aircraft that has both `assume_aircraft` and `entry_coordination` tasks (`l_twin_AE = 1`). Of those targets, 2,600 are assume, 924 are entry_coordination, and 11 are other types.
- **Temporal persistence.** Consecutive labelled samples within a session share the same target `task_name` 72.1% of the time in train and 76.5% in validation. Only about 68% of consecutive pairs in train are exactly 3 s apart, so labelled moments have gaps. The previous sample's target `task_name` is still present in the next sample's list about 80% of the time, so tasks can be aligned across snapshots by `(session, task_name)`.
- **List position.** In train, the target's mean relative position `row_index/(seq_len-1)` is 0.516, so there is no obvious positional prior. Treat any `row_index` signal with suspicion (see §8).
- **Doc discrepancy.** The Markdown's example submission shows `12797,9 6 4`, but the actual `sample_submission.csv` has `12797,9 6 41`. Trust the file.
- **Missing values.** An empty cell means "does not apply" or "not computed", never 0. Many `k_` columns are empty for aircraft not yet assumed. `distance_to_MD`/`time_to_MD` are empty when there is no conflict, `distance_to_zone` when no zone is crossed, and `exit_FL` when unknown.

## 3. Metric and submission

- **Metric:** top-3 hit rate. For each sample, take the first three **distinct** ids in `prediction`. The sample scores 1 if the labelled `row_index` is among them. The score is the mean over samples, each counting equally.
- **Leaderboards:** public = controller 020 (`020sc2`, `020sc3`, 915 samples). Private (decides the ranking) = controller 023 (948 samples). Controller 020 also flew `020sc1` in validation, so public-LB feedback is partially "seen controller" and is **not** a reliable proxy for the private score.
- **Reference scores (test):** random 0.094; `sample_submission.csv` baseline 0.352. The baseline ranks the most frequently labelled task types first, then within a type puts the aircraft closest to the sector boundary first.
- **Format:** CSV header `sample_id,prediction`. One line per test `sample_id`, exactly once. `prediction` holds space-separated integer `row_index` values, best first.
- **Scorer gaps you must cover yourself:** `scorer.py` does not check that ids belong to the sample (`0 ≤ id < seq_len`). It does not reject extra `sample_id`s, and it silently accepts fewer than three ids. Your own validator must check all of these.

## 4. Key modelling challenges

1. **Ranking within groups with exactly one positive.** Candidates compete only against others in the same `sample_id`. With exactly one positive, P(hit) = the sum of the true within-sample probabilities of the three chosen rows. So picking the top 3 by well-ordered within-sample scores is optimal. What matters is ordering, not global calibration. A listwise softmax objective matches this structure directly. Binary and pairwise objectives approximate it.
2. **Aircraft-level vs task-level features.** Most columns are identical across the tasks of one aircraft. Only `task_index`, `x_task_age`, and some `l_` columns separate an aircraft's tasks. The hardest case is assume vs entry_coordination twins.
3. **Generalisation to an unseen controller.** There are 16 training controllers. Controller strategies differ, scenarios repeat across controllers, and labels are noisy. Validation must hold out whole controllers.
4. **Strong autocorrelation.** Consecutive samples are near-duplicates, so the effective sample size is far below the sample count. Small metric differences are often noise.
5. **Informative missingness.** Whether a cell is NaN encodes state (for example "not assumed" or "no conflict"). Do not zero-fill blindly.
6. **Identifier traps.** `session`, `atco`, `timestamp`, `sample_id`, and raw callsigns can let a model memorise controller- or scenario-specific behaviour.

## 5. Stage 0 — Environment and data validation

1. Set up the environment (see §0.4). Record the Python and package versions, CPU count, and RAM. The machine had 8 cores and about 15 GB RAM, with about 6 GB free at writing time.
2. Load each CSV with explicit dtypes: `float32` for features, `int32`/`int16` for ids and flags, `category`/`string` for `session`/`task_name`/`timestamp`. Cache to Parquet/Feather after the first load.
3. Write `validate_data.py` and run it. It must assert and report:
   - the expected column sets per file, and that the test columns equal the train columns minus `target_index`;
   - rows, samples, sessions, and controllers per file, compared against §2 and the documentation. Report any discrepancy and do not hide it;
   - per sample: row count equals `seq_len`, `row_index` is exactly `0..seq_len-1`, `seq_len`, `session`, `atco`, `scenario`, and `timestamp` are constant within the sample, and `task_name` is unique;
   - labelled files: exactly one `target_index == 1` per sample;
   - `task_index` values, and consistency between `task_index` and the type prefix of `task_name`;
   - that aircraft-level columns are constant across rows of the same aircraft within a sample. Report which columns actually vary between twin tasks;
   - missingness rate per column, by `is_assumed`, and by split. Check for unexpected values (negative ages, out-of-range flags, `non_conformance_agreed_route_deviation_horizontally` > 100 km), duplicated rows, and timestamp ordering within sessions (including any midnight wrap);
   - session split disjointness (no session in two files) and controller overlap (020 appears in both validation and test, and that is expected);
   - that every test `sample_id` in `sample_submission.csv` exists in `test.csv`, and vice versa.

## 6. Leakage-safe validation protocol

1. **Metric module** (`metrics.py`): a vectorised top-3 hit rate that reproduces `scorer.py` exactly. It takes the first three distinct ids. Unit-test it against `scorer.py` on at least one generated validation submission, and require identical scores.
2. **Model selection: leave-controller-out CV on `train.csv` only.** Use `GroupKFold` with groups = `atco`, about 4 folds of 4 controllers each, so every fold covers all three scenarios. Report the mean, the std, and the per-fold scores. Hyper-parameters, feature sets, early-stopping rounds, smoothing coefficients, and ensemble weights are chosen **only** from this CV.
3. **Confirmation holdout: `validation.csv`.** Train on all of `train.csv` with the CV-selected configuration and score once per candidate on validation. Do not iterate many times against validation. If validation and CV disagree, report it and prefer the decision that is robust on both.
4. **Uncertainty.** Because of autocorrelation, estimate confidence intervals with a **session-level or contiguous-block bootstrap**, not an i.i.d. bootstrap. Compare models with paired per-fold differences.
5. **Adoption rule for any change:** mean CV improvement > 0, an improvement in at least 3 of 4 folds, and no material regression on validation. Otherwise keep the simpler model. Treat gains below about one fold-std as unproven.
6. **Grouping invariants:** all rows of a `sample_id` stay together, all samples of a session stay together, and all sessions of a controller stay in the same fold.
7. Report the samples, rows, sessions, and controllers in every train and eval split.
8. Never use public-leaderboard feedback for model selection.

## 7. Prioritised roadmap

Each stage ends with a logged row in `reports/experiments.csv` holding config, CV mean/std, per-fold scores, validation score, train time, and inference time.

- **Stage 0. Environment and data validation** (§5).
- **Stage 1. Metric and baselines.**
  - Random baseline on validation, averaged over seeds (expected to be near 3/mean `seq_len`).
  - Reimplement the documented rule baseline: type priority by target frequency in `train.csv`, then within a type the smallest `|distance_to_sector|` (also try `|time_to_sector|` / `l_rank_absf1_type`). The docs do not specify how ties between types are interleaved. Determine the exact rule by **reproducing `sample_submission.csv` on test features**. Report the exact-match rate per sample, then score the reproduced rule on validation and in CV.
  - Per-type prior baseline: score = positive rate of the type in the training folds.
- **Stage 2. Simple learned baseline.** A binary GBDT (LightGBM if available) on the 86 raw features plus `task_index` as a categorical. Use native NaN handling and no identifier columns. Rank by predicted score within the sample. This is the reference model.
- **Stage 3. First real submission.** Fit the Stage 2 model on `train.csv` and confirm on validation. Refit on train + validation with the number of rounds fixed from CV, scaled for the larger data. Generate and validate the submission (§15).
- **Stage 4. Ranking objectives**, each compared to Stage 2 on the same folds and features:
  - LightGBM `lambdarank` (`eval_at=[3]`, groups = sample sizes, data sorted by `sample_id`);
  - LightGBM `rank_xendcg` (listwise, softmax-like);
  - optionally a custom per-group softmax cross-entropy objective;
  - XGBoost `rank:pairwise` / `rank:ndcg` and CatBoost `YetiRank`/`QuerySoftMax`, only if installed;
  - for the binary model, row weights of `1/seq_len` or `1/(seq_len-1)` on negatives (because the metric weights samples, not rows).
- **Stage 5. Within-sample relative features** (§8.C). Ablate them as a group, then by subgroup.
- **Stage 6. Causal temporal features and smoothing** (§11). Only on top of the best Stage 4/5 model.
- **Stage 7. Robustness and ensembling.** Average 3–5 seeds. Ensemble complementary models (for example binary + lambdarank, or LightGBM + CatBoost) by **within-sample rank averaging** or z-scored score averaging, with weights from CV. Adopt only under the §6.5 rule.
- **Stage 8. Final model.** Refit the selected configuration on train + validation. Validation adds controllers 019 and 020; this is the documented, permitted use. Fix the iteration counts from CV, average seeds, predict test, and validate (§15).

## 8. Data preparation and feature engineering

### A. Base inputs

- Use the 86 documented feature columns plus `task_index` (categorical).
- **Exclude by default:** `sample_id`, `target_index`, `session`, `atco`, `timestamp`, and raw `task_name`.
- **Test only under leave-controller-out CV, with explicit justification:**
  - `scenario`;
  - `row_index` and `row_index/seq_len`. First investigate how AWARE orders the list (for example, grouped by aircraft). Keep them only if the ordering is a genuine inference-time property and the gain holds across folds;
  - `seq_len`.

### B. Missing values

- Keep NaN as NaN for GBDTs (native handling).
- Add explicit missingness indicators only where they add information beyond existing flags. Test, for example, `k_*` missing vs `is_assumed`, `exit_FL` missing, and `distance_to_zone` missing.
- Never fill with 0 where 0 is a meaningful value. `time_to_MD = 0` means "closest now".
- For linear or neural models, if any are used, impute per column with an indicator, and document it.

### C. Task-level and within-sample relative features

All are computed per `sample_id` from features only, with no labels.

- Type-relative: per-type counts already exist (`l_n_type`, `l_frac_type`). Add counts of other key types in the sample: number of transfer-ready aircraft, active conflicts, zone conflicts, and pending transfers-in (`x_transfer_in`).
- Urgency ranks within the sample and within the type: rank and normalised rank of `|time_to_sector|`, `time_to_MD`, `distance_to_zone`, `k_timeToExitPoint`, `x_task_age`, `x_state_age`, and `a_since_flight`. Also take the difference from the sample's minimum or median.
- Aircraft-level aggregates: the number of tasks for the aircraft and which task types it has. Example flags: has_assume, has_entry_coord, has_transfer, has_conflict.
- Twin-pair features for `l_twin_AE = 1`: the difference in `x_task_age` between the assume and entry_coordination tasks, and which is older, combined with `x_transfer_in`, `x_isc_notified`, `x_isc_coordinated`, `x_entry_coord_not_done`, and `time_to_sector`. Domain logic: coordination precedes the offer, which precedes the assume.
- Recency of the controller's attention: `a_is_last_cs`, `a_is_last2_cs`, `a_since_flight`, the rank of `a_since_flight` within the sample, and interactions with the task type (for example, the last input was an assume and the task is a transfer).
- State-consistency indicators per type. Examples: transfer task with `k_aircraftNeedsToBeTransferred`/`k_transferToolActivation`; exit_conditions task with `k_hasToClimbToExitFL`/`k_hasToDescendToExitFL`; zone_conflict task with `distance_to_zone`; conflict task with `time_to_MD`/`HMI_conflict_safety_alert`.
- Session-elapsed time: seconds since the session's **first** sample (causal). Test it carefully, because labelled moments may not start at the scenario start.

### D. Identity and high-cardinality columns

- Callsigns recur across sessions of the same scenario (same traffic). A callsign/scenario-level target encoding may transfer, or may memorise controller choices. Allow it **only** as an explicitly flagged experiment: out-of-fold encoding computed within the training folds, smoothed toward the type prior, evaluated under leave-controller-out CV, and adopted only under §6.5.
- Never use `atco` or `session` as features.

### E. Implementation rules

- Vectorise with `groupby(sample_id).rank/transform`. No Python loops over samples.
- Build features with one function shared by train, validation, and test. Freeze the feature list and order in an artifact and assert schema equality at inference.

## 9. Candidate models and selection criteria

| Model | Why | Input | Aggregation | Cost |
|---|---|---|---|---|
| Rule baseline | Reference, sanity check | Type prior + boundary distance | Sort within sample | Trivial |
| GBDT binary (LightGBM/XGBoost/CatBoost) | Strong on heterogeneous tabular data with NaN | Row features | Sort probabilities within sample | Low |
| GBDT lambdarank / pairwise | Optimises within-group order | Row features + group sizes | Sort scores | Low |
| GBDT listwise (`rank_xendcg`, softmax CE, `QuerySoftMax`) | Matches the one-positive structure | Same | Sort scores | Low |
| Rank-average ensemble | Variance reduction, complementary errors | OOF scores | Average within-sample ranks | Low–medium |

Do not add neural or sequence models unless the GBDT approaches plateau, time permits, and CV shows a gain. Use regularised settings suited to noisy, correlated data: larger `min_data_in_leaf`/`min_child_samples`, feature and bagging fractions around 0.7–0.9, a modest number of leaves, and early stopping on a **custom top-3 feval** computed on the held-out fold. Keep tuning budgets small, and tune on CV only.

**Selection criterion:** the highest mean leave-controller-out CV top-3 score that satisfies §6.5. Ties go to the simpler and faster model.

## 10. Group-wise ranking and top-3 output

- Score every candidate row. Within each `sample_id`, sort by descending score. Break ties deterministically by ascending `row_index`.
- Output the first three **distinct** `row_index` values. `seq_len ≥ 9` guarantees three exist. Assert this anyway.
- Twin-aware post-processing is allowed only if error analysis shows twin confusions dominate failures, it is tuned on CV only, and it passes §6.5. By §4.1 it can help only by correcting score miscalibration.

## 11. Optional temporal experiments (strictly causal)

Preconditions: verify that samples can be ordered by `(session, timestamp)` and aligned across snapshots by `(session, task_name)`. Measure and report the alignment rate and the gap distribution.

Candidate features for snapshot *t* use only snapshots *< t* in the same session:

- presence of the same `task_name` in the previous snapshot, the run length of consecutive presence, and the time gap to the previous snapshot;
- deltas of `distance_to_sector`, `time_to_sector`, `x_task_age`, `a_since_flight`, `time_to_MD`, and `a_n60` versus the previous snapshot;
- a "new input on this aircraft since the last snapshot" indicator, and a new-task indicator;
- **previous model scores (two-stage):** the stage-1 score or rank of the same task at *t-1*. Training data must use **out-of-fold** stage-1 scores, so the train and test distributions match. At test time, use stage-1 test predictions of earlier snapshots only;
- **causal score smoothing:** `s'_t = s_t + α · Σ_k decay^{Δt_k} · s_{t-k}` for the same `task_name`, with α and decay tuned on CV only.

Never use true labels of earlier snapshots as features: they are unavailable at test time. Never use later snapshots. The 72–77% label persistence is a property of the labels, not a usable feature. Keep the non-temporal model as the fallback, and adopt temporal additions only under §6.5.

## 12. Error analysis

For the best model and the baseline, on CV out-of-fold predictions and on validation, report:

- top-1/top-3 hit rate overall, and by true task type;
- by session, controller, and scenario;
- by `seq_len` bucket;
- for twin_AE samples vs others, with the twin confusion rate: the true task is assume and the model ranks entry_coordination of the same aircraft higher, or the reverse;
- for samples with multiple tasks on the target aircraft (`l_n_flight > 1`);
- for conflict types (0, 1, 6) and coordination/hand-over types (2, 3, 4, 5);
- the rank distribution of the true task, and how many misses had it at rank 4–5 (near misses) vs deep misses;
- a confusion summary of the predicted top-1 type vs the true type;
- feature importance (gain and permutation on CV folds) and the stability of top features across folds;
- train time, inference time, and peak memory;
- seed variance of the CV score across at least 3 seeds.

Use these results to choose the next targeted experiment. Write a one-line hypothesis before running each one.

## 13. Efficiency and reproducibility

- Fix seeds (`numpy`, model `seed`/`random_state`, `deterministic=True` where supported) and log them.
- Keep all hyper-parameters, feature lists, and paths in a YAML/JSON config. The config hash and git commit, if any, go into each experiment log.
- Cache loaded data and features as Parquet/Feather. Avoid recomputing.
- Use `float32` features, data sorted by `sample_id` (required for ranking groups), and `n_jobs` up to the available cores. Expect seconds to minutes per fit on about 475k rows; flag anything slower.
- Save model artifacts, the feature schema, and the selected iteration counts. Inference loads the artifacts and does not retrain.
- Handle errors explicitly: fail loudly on schema mismatch, unexpected NaN in identifiers, groups not matching `seq_len`, or invalid predictions.

## 14. Code architecture and deliverables

```
config/default.yaml
src/
  io.py               # typed loading, caching, schema checks
  validate_data.py    # Stage 0 audit -> reports/data_audit.md
  metrics.py          # top-3 hit rate (matches scorer.py), block bootstrap
  cv.py               # leave-controller-out folds, fold reports
  baselines.py        # random, rule (reproduces sample_submission), type prior
  features.py         # base + relative + optional temporal (causal) features
  models.py           # binary / lambdarank / xendcg / optional XGB, CatBoost wrappers
  rank.py             # within-sample ranking, top-3 extraction, ensembling
  error_analysis.py   # slices from §12
  train.py            # CV experiments + final fit; logs to reports/experiments.csv
  predict.py          # artifacts -> submission (no training code paths)
  validate_submission.py
reports/  data_audit.md, experiments.csv, error_analysis.md, REPORT.md
artifacts/ models, feature schema, config snapshot
submissions/ validation_submission.csv, submission.csv
requirements.txt, README.md (exact commands to reproduce end to end)
```

`REPORT.md` must contain:

- the data audit summary and discrepancies;
- the baseline reproduction result (match rate vs `sample_submission.csv`; validation and CV scores);
- the experiment table with mean ± std, per-fold scores, and validation scores;
- the selected model and **why**, with the evidence;
- the error analysis;
- limitations, unverified assumptions, and next steps.

## 15. Final submission generation and validation

`validate_submission.py` must check, and fail on any violation, that:

- the header is exactly `sample_id,prediction`;
- the number of rows equals the number of unique test `sample_id`s (1,863 expected);
- the `sample_id` set exactly equals the test set: none missing, none extra, none duplicated;
- each prediction has three space-separated integer tokens, all distinct, each in `[0, seq_len-1]` of that sample;
- `prediction` is written as a string, not a float.

Also:

- Run `python scorer.py --labelled validation.csv submissions/validation_submission.csv`, using a model trained **without** validation data, and record that score next to your internal metric. They must match.
- Spot-check five random test samples by mapping the predicted `row_index` values back to their `task_name`s.
- Never assume a file is valid because it was written. Re-read it from disk and validate it.

## 16. Acceptance criteria

- [ ] Stage 0 audit run, with counts compared to the documentation and discrepancies reported.
- [ ] Internal metric matches `scorer.py` exactly on a validation submission.
- [ ] Rule baseline reproduced (match rate vs `sample_submission.csv` reported) and scored on validation and CV.
- [ ] At least one learned model beats the rule baseline on leave-controller-out CV **and** on validation, with the margin and uncertainty reported.
- [ ] Binary vs ranking objectives compared on identical folds and features.
- [ ] Every adopted feature group, temporal addition, or ensemble passes the §6.5 rule. Rejected experiments are logged too.
- [ ] No validation or test labels used outside the documented final refit. All features causal. No identifier memorisation without an explicit, validated experiment.
- [ ] Final `submission.csv` passes `validate_submission.py`. Pipeline reproducible from `README.md` with fixed seeds.
- [ ] `REPORT.md` explains the final choice, its limitations, and next steps. Every number in it traces to a logged run.
