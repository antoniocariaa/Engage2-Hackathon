# Prompt: fix `exit_coordination` in the AWARE task ranker

## 1. Goal

You work in the existing repository `/home/faizan/Desktop/Engage`. It contains a working pipeline for the AWARE active-task challenge. `aware_task_prediction_organized.md` describes the challenge. `README.md` and `reports/REPORT.md` describe the pipeline and every experiment run so far. The model scores every candidate task in a snapshot, ranks them within the snapshot, and is evaluated by top-3 hit rate.

Your job is to raise the **overall** top-3 hit rate by fixing the class `exit_coordination` (`task_index` 3), the largest pool of misses. A change that raises exit-coordination recall but lowers overall top-3 is a failure.

Do not follow `IMPLEMENTATION_PROMPT.md`. The rules for this task are in section 4.

## 2. Current state

These numbers come from a read-only analysis of the current model. Re-verify them in step 0.

The reference model is an equal-weight within-sample z-score blend of `lgb_binary_3s` and `lgb_lambdarank_3s`: LightGBM, 3 seeds each, 86 features plus `task_index`. Its out-of-fold and validation scores are in `cache/preds/eq_lgb_binary_3s+lgb_lambdarank_3s.npz`, keys `oof` and `val`. Rows align with `cache/train.parquet` and `cache/validation.parquet`, both sorted by `sample_id`, `row_index`.

| Quantity | CV, 4 controller folds | Validation |
|---|---|---|
| Overall top-3 | 0.6525 | 0.6589 |
| Exit-coordination targets | 704 of 11,923 samples (5.9%) | 41 of 1,252 |
| Positive rate per type-3 row | 1.3% | 0.8% |
| Top-3 when the truth is exit coordination | 0.037 | 0.146 |
| Median overall rank of the true row | 11 | 11 |
| True row is the best-scored type-3 row | 41.5% | 34.1% |
| True row is among the 3 best type-3 rows | 85.4% | 56.1% |
| Overall top-3 if the true type were known for every sample | 0.970 | 0.962 |
| Overall top-3 if the true type were known only for exit-coordination samples | 0.7007 | 0.6725 |

**Diagnosis.** The failure is at the type level. The model rarely believes that the controller is coordinating an exit right now. When the type is known, choosing the right aircraft mostly works. In CV, when the truth is exit coordination, the predicted top-1 is a transfer in 316 and an assume in 232 of the 704 cases. The exit-coordination-only oracle bounds the gain from this task at about +0.048 CV.

**Already measured, does not work:**

- A constant bias on type-3 scores always lowers overall top-3. In CV, +0.25 z gives 0.6509, +0.5 gives 0.6426, +1.0 gives 0.5779. Do not use biases, per-type thresholds, or upweighting of type-3 positives as the fix. What is missing is information about *when* coordination happens, not calibration.
- Causal score smoothing and pooling scores across neighbouring snapshots give no gain.
- The generic feature groups `rel`, `ac`, `temp`, `act`, `typed` and target encoding give no gain. See `reports/experiments.csv`.
- A generic future signal, the time until a task disappears from the list, lifts overall CV to 0.6962. It lifts exit coordination only from 0.037 to 0.057, because it mainly helps transfer and assume. Exit coordination needs its own signals.

**Signals on type-3 rows, target vs other type-3 rows:**

| Signal | Train: target / other | Validation: target / other | Consistent |
|---|---|---|---|
| Median `a_since_flight`, s | 212 / 321 | 188 / 291 | yes |
| `a_last_assign_level_clearance` | 0.44 / 0.34 | 0.49 / 0.33 | yes |
| `k_outCoordMismatch` | 0.66 / 0.55 | 0.83 / 0.57 | yes |
| Coordination state of the aircraft changes in the next 30 s | 0.099 / 0.037 | 0.122 / 0.042 | yes |
| Same, in the previous 30 s | 0.124 / 0.105 | 0.098 / 0.116 | no signal |
| `k_coordinationToolActivation` | 0.49 / 0.30 | 0.00 / 0.30 | no |
| `a_is_last_cs` | 0.14 / 0.04 | 0.00 / 0.03 | no |
| `k_exitConditionsAreMet` | 0.34 / 0.56 | 0.57 / 0.50 | no |

"Coordination state" means any of `k_outCoordRevision`, `k_out_COORDINATED`, `k_out_NOTIFIED`, `k_outCoordMismatch`, compared at the aircraft's next or previous snapshot.

**Other facts:**

- For exit-coordination targets in train, the controller's next input on the aircraft within 60 s is: none 46%, a level clearance 26%, a lateral clearance 27%.
- Only 13% of exit-coordination target tasks disappear from the list within 30 s. For transfer and assume targets it is 95–97%.
- Across the 16 training controllers, exit coordination is between 1.1% and 11.1% of a controller's targets. Expect controller-style noise. Small gains may not be real.
- Samples come in blocks of consecutive snapshots 3 s apart, with a median of 3 snapshots per block. Within a block the target is the same 95% of the time, and 87% of target changes happen at block boundaries. Block boundaries are visible in test as well.

## 3. Data mechanics

- **Callsign of a row:** strip the task-type prefix (`TASK_TYPES` in `src/data_io.py`, plus `_`) from `task_name` and take the first `_`-separated token. `features.aircraft_key` already does this.
- **Snapshot time:** `features.seconds(timestamp)`. Order snapshots by session, then seconds.
- **Controller-input timeline per (session, callsign):** at every snapshot where `a_since_flight < 3600`, the input time is the snapshot seconds minus `a_since_flight`, and the input type comes from the `a_last_*` flags at that snapshot. Deduplicate on (session, callsign, input time). Use `pd.merge_asof` by session and callsign, forward or backward, to get the time to the next input and since the previous one.
- **State timelines** for `k_outCoordRevision`, `k_out_COORDINATED`, `k_out_NOTIFIED`, `k_outCoordMismatch`, `exit_FL`, `k_clearedFL`: keep one row per (sample, aircraft), sort by time, and mark a change when the value differs from the aircraft's previous or next snapshot. Snapshots exist only at labelled moments, so always also emit the time gap to the snapshot where the change was observed.
- **Task lifetime:** the last time the (session, `task_name`) pair appears. Use NaN if the task is still present at the session's last snapshot.
- **Coverage:** train, validation and test have the same structure. In all three, 95–96% of rows have a known disappearance time.

## 4. Rules

1. **Labels.** Never use validation or test labels for features, training, tuning or post-processing. The only exception is the final refit on train plus validation.
2. **Future snapshots are allowed.** Features may use earlier and later snapshots of the same session, because `test.csv` contains whole sessions. Put every forward-looking feature in a group whose name says so, for example `xc_lead`, so it can be switched off. `run_final` in `src/train.py` (around line 205) refuses specs marked `noncausal`. Replace that guard with an explicit opt-in flag instead of deleting it.
3. **Validation.** Use the existing 4 leave-controller-out folds from `src/cv.py` for every decision, and `validation.csv` as a one-shot confirmation. Use seeds 42, 43 and 44 for anything you compare. Any learned stacking input, such as stage-A probabilities, must be out-of-fold by controller for training rows.
4. **Adoption.** Adopt a change only if overall CV top-3 improves on the mean and in at least 3 of 4 folds, and validation drops by no more than 0.01. Log rejected runs too.
5. **Reporting.** For every run, report overall CV mean ± std, the per-fold scores, validation, and top-3 by true type for types 3, 5, 4, 12, 2 and 10.
6. **Code.** Do not change the behaviour of existing experiments or feature groups. Add new ones. Log every run through `src.explog.log_experiment` into `reports/experiments.csv`.
7. **Compute.** The machine has 8 cores and about 15 GB RAM. Other training jobs may be running. Check `ps` first, never kill them, and size `n_jobs` accordingly.
8. **Time.** The competition closes within hours. Check the remaining time first. `submissions/submission.csv` must stay valid at every moment. Copy it to a backup before replacing it.
9. **Evidence.** Every number you report must come from code you ran.

## 5. Plan

Work in this order. Stop a step early if it clearly fails, and write a one-line hypothesis before each run.

**Step 0: reproduce the diagnosis (10 minutes at most).** From the reference out-of-fold scores, recompute overall and per-type top-3, the exit-coordination rank statistics, and both oracles in section 2. If your numbers differ, stop and report.

**Step 1: exit-coordination features (main step).** Add a causal group `xc` and a future group `xc_lead`. Values are filled on type-3 rows and NaN elsewhere, unless noted. Hypothesis: coordination happens when the agreed exit level no longer matches what the aircraft is cleared to, and it shows up as a coordination-state change shortly afterwards.

- Causal, in `xc`:
  - `exit_FL − k_clearedFL` and `exit_FL − current_FL`, their absolute values, and a flag for "cleared level differs from agreed exit level";
  - `k_levelClearanceAgeS`;
  - `k_outCoordMismatch` combined with `k_exitConditionsAreMet`;
  - time since the last change of the coordination state and of `k_clearedFL`;
  - ranks among the sample's type-3 rows of `a_since_flight`, `k_levelClearanceAgeS` and `k_timeToExitPoint`;
  - sibling tasks of the same aircraft: has an exit-conditions, transfer or conflict task;
  - on all rows: the sample's count of type-3 rows with a mismatch, and its count of competing ready work, meaning assume tasks with `x_transfer_in = 1` and transfer tasks with `k_aircraftNeedsToBeTransferred = 1`.
- Future, in `xc_lead`:
  - time to the next change of each coordination-state column, of `exit_FL` and of `k_clearedFL` for the aircraft, plus the gap to the snapshot where the change was observed;
  - time to the next controller input on the aircraft, and its type;
  - time until the type-3 task disappears from the list;
  - time until the end of the current block.

Run binary and lambdarank with 3 seeds each, first on base + `xc`, then on base + `xc` + `xc_lead`. Rebuild the equal z-blend and compare it with the reference.

**Step 2: two-stage type model.** Do this only if step 1 leaves a large gap to the oracle. Stage A is a sample-level LightGBM multiclass model that predicts the true task type. Its inputs are per-type aggregates of the sample: for each type, the row count, the min and max of its key urgency features, the max of its future signals, plus the sample's `a_n60` and `a_n300`. Train stage A out-of-fold by controller, nested inside each outer fold for training rows. Then either feed `P_A(type of this row)` and its log to the ranker as row features, or combine `log P_A(type)` with the log of the within-type softmax of the ranker score. Compare both. The all-types oracle is 0.970, so stage A can help beyond exit coordination too.

**Step 3: within-type ranking.** Do this only if steps 1 and 2 raise type-level recall. 85% of exit-coordination truths are already among the 3 best type-3 rows, so this has low priority.

**Do not:** add constant biases or per-type thresholds, upweight type-3 positives, memorise callsigns or scenarios, or tune anything on validation.

## 6. Deliverables

- **Code:** new groups in `src/features.py`, new entries in `EXPERIMENTS` in `src/train.py`, and optionally `src/stage_a.py`.
- **Report:** `reports/exit_coordination.md`, with one table holding the reference and every run (overall CV mean ± std, per fold, validation, per-type top-3), plus the hypothesis and the decision for each run.
- **Submission, only if something is adopted:**
  1. Refit each blend component with `python -m src.train final <experiment>`.
  2. Write the validation submission from the train-only models with `python -m src.predict <a>,<b> --val-from-cv` and score it with `python scorer.py --labelled validation.csv submissions/validation_submission.csv`.
  3. Write the test submission with `python -m src.predict <a>,<b> --out submissions/submission.csv`.
  4. Check it with `python -m src.validate_submission submissions/submission.csv`.
- **Final note:** what was adopted, the measured gain overall and on exit coordination, and what was left out and why.
