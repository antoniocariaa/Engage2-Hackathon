# AWARE active-task prediction: final report

## Update (12:42): final model now uses later snapshots of each session

**`submissions/submission.csv` is now `lr_rw5_lead_3s`.** It is LightGBM lambdarank with 3 seeds (42–44), trained on the 86 base features + `task_index` plus the forward-looking group `lead_blk`. Training samples are weighted by 1/√(length of the run of identical consecutive labels). It was refit on train + validation with R_final = 258.

**This is a deliberate change of rules.** The user asked for at least 0.80, and `EXIT_COORDINATION_PROMPT.md` allows future snapshots. Rule 6 below (causal features only) therefore no longer applies to the final model. `lead_blk` uses no labels; it reads only timestamps, task names and `a_since_flight`. It is computed per session from the file being scored, and `test.csv` contains whole sessions. **The result is a transductive, not real-time, model.** The previous causal submission is kept as `submissions/backup_submission_lambdarank_rw5.csv` (CV 0.661, val 0.661).

`lead_blk` features:
- **Block structure:** blocks of consecutive snapshots ≤ 3 s apart, with position and time to block end.
- **Task lifetime relative to the block:** when the task was last seen, and when it was first seen missing.
- **The next controller input on the aircraft relative to the block,** and its type.

Strongest signals (validation, descriptive):
- The task is last seen exactly at the block end for 49% of targets vs 3% of other rows.
- An input on the aircraft falls inside the block for 39% of targets vs 4% of others.

| model | CV mean ± std | folds | validation (`scorer.py`) |
|---|---|---|---|
| rule baseline | 0.356 ± 0.028 | .390/.362/.324/.347 | 0.4848 |
| lgb_lambdarank_rw5_3s (best causal) | 0.661 ± 0.023 | .682/.679/.645/.637 | 0.6613 |
| **lr_rw5_lead_3s (final)** | **0.901 ± 0.027** | **.901/.938/.890/.875** | **0.9289** (session-bootstrap CI [0.912, 0.939]) |
| + block pooling of scores (β = 1, tuned on CV) | 0.902 | .903/.938/.893/.875 | 0.9225 | 

Block pooling was **rejected**: only 2/4 folds improved and validation fell by 0.006.

Supporting checks:
- **Run-length weighting (causal)** was adopted before this. Against unweighted lambdarank it was better in 4/4 folds, with validation +0.014 (CI [+0.010, +0.021]).
- **XGBoost on the GPU (MX450)** was about 20× faster than on the CPU. XGBoost pairwise reached CV 0.655 and did not beat LightGBM.
- **CatBoost** remained too slow, even on the GPU.

CV top-3 by true type for the final model:

| type | top-3 |
|---|---|
| transfer | 0.983 |
| assume | 0.976 |
| zone_conflict | 0.921 |
| non_conformance | 0.930 |
| conflict_resolution | 0.897 |
| qos | 0.869 |
| return_to_route | 0.843 |
| exit_conditions | 0.840 |
| exit_conflict | 0.839 |
| entry_coordination | 0.707 |
| entry_conflict | 0.645 |
| **exit_coordination** | **0.565** (weakest) |

Top-1 is 0.790 in CV and 0.793 on validation. The twin confusion rate is 3.9%. The top gain features are `L_last_minus_b_end`, `L_next_input_minus_b_end`, `L_last_minus_T` and `task_index`.

Reproduce the final model:

```bash
$PY -m src.train cv lr_rw5_lead_3s
$PY -m src.train final lr_rw5_lead_3s
$PY -m src.predict lr_rw5_lead_3s --val-from-cv && $PY scorer.py --labelled validation.csv submissions/validation_submission.csv
$PY -m src.predict lr_rw5_lead_3s --out submissions/submission.csv
$PY -m src.validate_submission submissions/submission.csv
```

### Update (12:51): blend with block-boundary state-change features

- **New group `lead_chg`** (future, opt-in). It records the aircraft's state at its first snapshot after the block and its last snapshot before the block, relative to the current snapshot, for 20 state columns (coordination flags, revision, cleared and exit levels, assume/transfer status, and so on). It also records whether the task still exists right after the block.
- **`lr_rw5_leadchg_3s` (lead_blk + lead_chg):** CV 0.9090 ± 0.023 (.907/.940/.905/.884), +0.0079 vs `lr_rw5_lead_3s` in 4/4 folds. Validation was 0.9177 (−0.011), so on its own it fails the adoption rule.
- **Equal z-blend of `lr_rw5_lead_3s` + `lr_rw5_leadchg_3s` (weights fixed in advance): CV 0.9058 (+0.0047, CI [+0.001, +0.008], 4/4 folds), validation 0.9297 (`scorer.py`).** Adopted as `submissions/submission.csv`. The previous 0.94-public-LB file is kept as `submissions/backup_submission_lead.csv`.
- **Where the remaining misses are:** 35% of training targets remain in the task list more than 30 s after their block ends. Those score 0.78 top-3, against 0.965 for the rest. They are mostly exit/entry coordination, zone conflict and return-to-route.

The sections below describe the earlier causal-only work. They are kept as the experiment history.

---

**Selected model:** an equal-weight blend of two LightGBM models on the 86 documented features plus `task_index` (categorical). The blend averages within-sample z-scores; the binary model and the lambdarank model each average 3 seeds. Both models were refit on train + validation with iteration counts fixed from CV. The output is `submissions/submission.csv`.

| | leave-controller-out CV (4 folds, train only) | validation (019, 020; train-only fit) |
|---|---|---|
| Rule baseline (= `sample_submission.csv`) | 0.3560 ± 0.0278 | 0.4848 |
| **Selected blend** | **0.6524 ± 0.0252** (per fold 0.6688 / 0.6788 / 0.6346 / 0.6275) | **0.6589** (`scorer.py`: 0.6589) |
| Margin over rule (paired, session-block bootstrap 95% CI) | +0.2966 [+0.2752, +0.3179], better in 4/4 folds | +0.1741 [+0.1534, +0.2030] |

Every number below comes from code run in this session. The sources are `reports/experiments.csv` (all runs, including rejected ones), `reports/comparisons.csv` (paired §6.5 checks), `reports/data_audit.md`, `reports/error_analysis.md`, and `reports/importance.md`.

## 1. Data audit (Stage 0)

`python -m src.validate_data` writes `reports/data_audit.md`.

- **Documented counts all match.** Train has 34 sessions, 16 controllers, 11,923 samples, 432,578 rows and 96 columns. Validation has 3 sessions, 1,252 samples and 42,110 rows. Test has 5 sessions, 1,863 samples, 66,409 rows and 95 columns. Test columns equal train columns minus `target_index`. There are 86 feature columns (x_ 8, k_ 32, a_ 12, l_ 9, 25 others).
- **Per-sample invariants hold in all splits.** Rows per sample equal `seq_len`. `row_index` runs exactly 0..seq_len−1. Session, atco, scenario and timestamp are constant within a sample, and `task_name` is unique. Every labelled sample has exactly one positive. The `task_name` prefix always matches `task_index`. Type 11 never occurs.
- **Timestamps are strictly increasing** within every session, with no midnight wrap. Sample ids are contiguous per session. Sessions are disjoint across files. The only controller overlap is 020, which appears in both validation and test, as documented.
- **Only a few columns differ between an aircraft's tasks:** `x_task_age`, `l_n_type`, `l_frac_type`, `l_rank_age_type`, `l_rank_f0_type`, `l_rank_f1_type` and `l_rank_absf1_type`. For assume/entry-coordination twins, `x_task_age` differs in 94% of pairs.
- **Descriptive label facts:** consecutive targets are the same task 72.1% of the time in train and 76.5% in validation. The mean relative target position is 0.516. Only 67.6% of consecutive train samples are exactly 3 s apart.
- **Discrepancies found:**
  1. Negative `k_levelClearanceAgeS` / `k_headingClearanceAgeS` values in all splits: 158/161 rows in train, 12/6 in validation, 19/13 in test. They were left as is, because trees are robust to them.
  2. 365 train rows and 36 test rows have a horizontal route deviation above 100 km. These are the documented likely errors and were left as is.
  3. The docs' example submission row `12797,9 6 4` is really `9 6 41` in the file.
  4. **`FEATURES.md` is not in the workspace**, so §8 of `aware_task_prediction_organized.md` served as the column dictionary.
  5. The docs describe the baseline as "most frequently labelled types first". The actual rule orders types by **positive rate** (see §2).

## 2. Baselines (Stage 1) and metric check

- `python -m tests.test_metric_vs_scorer` showed the internal metric equals `scorer.py` on three validation submissions (rule, random and all-ties). The vectorised path, the parsed-submission path and the fast feval path agree to 1e-12.
- **Exact reproduction of `sample_submission.csv`: 1863/1863 test samples (100.0%) match exactly.** The rule ranks types by training positive rate (5 > 7 > 13 > 4 > 12 > 6 > 1 > 9 > 3 > 10 > 8 > 2 > 0), then by ascending `|time_to_sector|` within a type (equivalently `l_rank_absf1_type`), then by ascending `row_index`. Ranking types by raw positive count matches only 55.5%. Using `|distance_to_sector|` within type matches 94.1%.
- Scores (CV / validation):

| baseline | CV | validation |
|---|---|---|
| random (20 seeds on validation, 5 seeds in CV) | 0.0908 | 0.0969 (3 / mean seq_len = 0.0892) |
| type prior | 0.3036 | 0.3906 |
| rule | 0.3560 | 0.4848 |

  The rule's validation score (0.485) is far above its stated test score (0.352) and its CV score (0.356). Validation (019, 020) is an easier or different holdout than the CV folds.

## 3. Protocol

- **Folds:** `GroupKFold` on `atco` gives 4 folds of 4 controllers each: {2,3,10,18}, {7,9,14,15}, {4,6,11,16} and {1,5,8,17}. Every fold contains all 3 scenarios. Sizes per fold are in `reports/folds.csv`; eval folds hold 2,941–3,039 samples and 8–9 sessions.
- **Iteration count:** each fold records the top-3 curve on its held-out fold, with early stopping as a bound. A single round count R is the argmax of the smoothed mean curve across folds. All reported CV scores are at that common R.
- **Validation:** one train-only fit uses R scaled by the data size. The final refit on train + validation scales R again.
- **Adoption rule (§6.5):** the paired CV difference must be > 0, at least 3 of 4 folds must improve, and validation must not regress by more than 0.01. Session-block bootstrap gives the CIs.
- **Seed noise:** the reference binary model gives CV **0.6496 / 0.6445 / 0.6459** for seeds 42/43/44, and single folds move by up to ±0.03 between seeds. Differences of a few thousandths between single-seed runs are therefore noise. The final decision was made on 3-seed models.

## 4. Experiments

All runs use the same folds. "Δ" is the paired difference against the reference named in the row.

| stage | experiment | CV mean ± std | folds (0/1/2/3) | val | vs reference: Δ CV [CI], folds better, Δ val | decision |
|---|---|---|---|---|---|---|
| 2 | lgb_binary (reference, seed 42) | 0.6496 ± 0.0245 | .6698/.6667/.6450/.6168 | 0.6518 | — | reference; first submission |
| 4 | lgb_lambdarank | 0.6577 ± 0.0274 | .6794/.6821/.6430/.6265 | 0.6414 | +0.0082 [−0.0003,+0.0167], 3/4, −0.0104 | rejected (val) |
| 4 | lgb_binary_w (neg weight 1/(n−1)) | 0.6525 ± 0.0252 | .6719/.6765/.6339/.6278 | 0.6502 | +0.0030 [−0.0034,+0.0098], 3/4, −0.0016 | within seed noise; not adopted |
| 4 | lgb_softmax (custom per-sample softmax CE) | 0.6502 ± 0.0259 | .6675/.6772/.6285/.6275 | 0.6398 | +0.0007, 2/4, −0.0120 | rejected |
| 4 | lgb_xendcg | 0.6440 ± 0.0250 | .6712/.6591/.6264/.6194 | 0.6302 | −0.0055, 2/4, −0.0216 | rejected |
| 4 | xgb_binary | 0.6460 ± 0.0223 | .6586/.6709/.6264/.6278 | 0.6510 | — | not better |
| 4 | xgb_pairwise | 0.6472 ± 0.0235 | .6654/.6693/.6302/.6238 | 0.6398 | — | not better |
| 4 | cat_querysoftmax | — | — | — | — | aborted: over 17 min, time-boxed |
| 5 | + rel (within-sample ranks, sample type counts) | 0.6371 ± 0.0226 | .6494/.6627/.6207/.6157 | 0.6494 | −0.0123 [−0.0205,−0.0037], 0/4, −0.0024 | rejected |
| 5 | + ac (aircraft task-set, age vs. siblings, twin age diff) | 0.6480 ± 0.0327 | .6712/.6798/.6288/.6121 | 0.6326 | −0.0015, 2/4, −0.0192 | rejected |
| 5 | + rel + ac | 0.6409 ± 0.0260 | .6532/.6716/.6200/.6188 | 0.6534 | −0.0086 [−0.0164,−0.0004], 2/4, +0.0016 | rejected |
| 6 | + temp (causal deltas vs. previous snapshot) | 0.6464 ± 0.0239 | .6576/.6749/.6275/.6258 | 0.6502 | −0.0030, 2/4, −0.0016 | rejected |
| 6 | causal score smoothing (α × half-life grid) | best α = 0 | — | — | every α > 0 lowers CV | rejected |
| 8.D | (scenario, task_name) out-of-fold target encoding | 0.6444 ± 0.0208 | .6454/.6726/.6362/.6235 | 0.6685 | CV −0.005, val +0.017 | rejected (CV decides) |
| tune | num_leaves 15 / 7 / 63, ff 0.5 + λ 10, extra-trees, l15 + ff5 | 0.6326–0.6527 | see log | 0.616–0.647 | none passes §6.5 | base params kept |
| abl. | drop a_since_any, a_n60, a_n300 (3 seeds) | 0.6419 ± 0.0219 | .6552/.6657/.6254/.6214 | 0.6446 | −0.0029, 1/4, −0.0112 | rejected |
| 7 | lgb_binary_3s (seeds 42–44) | 0.6448 ± 0.0238 | .6627/.6680/.6251/.6235 | 0.6558 | — | blend component |
| 7 | lgb_lambdarank_3s | 0.6580 ± 0.0245 | .6794/.6785/.6420/.6322 | 0.6470 | vs binary_3s: +0.0132 [+0.0044,+0.0213], 4/4, −0.0088 | blend component |
| 7 | **equal z-blend binary_3s + lambdarank_3s** | **0.6524 ± 0.0252** | .6688/.6788/.6346/.6275 | **0.6589** | vs binary_3s: +0.0076 [+0.0029,+0.0127], 4/4, +0.0032 | **selected** |
| 7 | equal z-blend + xgb_pairwise | 0.6536 ± 0.0246 | .6719/.6775/.6356/.6295 | 0.6486 | vs binary_3s: +0.0088, 4/4, −0.0072 | not adopted |
| 7 | CV-weighted blend (0.1 binary / 0.9 lambdarank, seed-42 models) | 0.6590 | .6817/.6834/.6457/.6251 | 0.6414 | ≈ lambdarank alone | not adopted |

Speed: one LightGBM CV run takes 30–115 s on 8 threads, a full-train fit takes 7–35 s, and inference on validation takes under 0.35 s. CV processes peaked at about 1.4–2.0 GB RSS (observed with `ps`).

## 5. Why this model

1. **Every learned model beats the rule by about +0.29 CV and about +0.17 validation.** The binary LightGBM reference alone was the first validated submission.
2. **Lambdarank against binary (3 seeds) is a CV/validation disagreement.** CV favours lambdarank by +0.013, with the paired CI excluding 0 and all 4 folds better. Validation favours binary by 0.009, with a CI that includes 0 and covers only 2 controllers.
3. **The blend is the choice that holds on both.** §6.3 asks for the decision that is robust on both CV and validation. The blend's weights were fixed in advance at 0.5/0.5, so they were tuned on nothing. It beats the binary reference on CV in 4/4 folds, with CI [+0.003, +0.013], **and** on validation (+0.003). Lambdarank alone has the higher CV (+0.0055 over the blend) but is 0.012 lower on validation, with CI [+0.002, +0.029] in the blend's favour.
4. **Adding XGBoost pairwise is not worth it.** It raises CV only marginally and lowers validation, so the simpler two-model blend wins.
5. **The base features stay unchanged.** Every engineered group was tested under the same protocol, and none improved leave-controller-out CV. Within-sample rank features were significantly worse (0/4 folds, CI below 0). With only 16 training controllers and noisy labels, extra features mostly let the model fit controller-specific patterns. The documented `l_` columns already carry the useful within-sample ranks.

## 6. Error analysis (selected blend; `reports/error_analysis.md`)

- **Overall:** CV OOF top-1 is 0.355 and top-3 is 0.6525, with a session-bootstrap CI of [0.629, 0.675]. Validation top-1 is 0.355 and top-3 is 0.6589, with CI [0.614, 0.742].
- **Misses (CV):** of 4,143 misses, 1,327 are near misses with the truth at rank 4–5, 1,568 have it at rank 6–10, and 1,248 are deep misses beyond rank 10.
- **CV top-3 by true type:**

| type | top-3 | targets |
|---|---|---|
| transfer | 0.876 | 3,145 |
| assume | 0.826 | 3,003 |
| zone_conflict | 0.744 | 1,950 |
| non_conformance | 0.709 | 86 |
| return_to_route | 0.474 | 881 |
| conflict_resolution | 0.447 | 351 |
| entry_coordination | 0.332 | 940 |
| exit_conditions | 0.179 | 407 |
| exit_conflict | 0.113 | 186 |
| qos | 0.101 | 237 |
| **exit_coordination** | **0.037** | 704 |
| entry_conflict | 0.032 | 31 |

  The model has essentially learned "who needs handing over or clearing now". Coordination tasks for future or exiting traffic, and qos tasks, are rarely predicted.
- **Twins:** twin_AE samples score 0.689 top-3 vs 0.637 for others in CV. On validation it is the reverse: 0.593 vs 0.691. The twin confusion rate is 14.8% (CV) and 15.1% (validation), against 26.2% and 31.6% for the rule baseline. Twin confusions do not dominate the failures, so no twin post-processing was applied (§10).
- **Aircraft with several tasks:** samples whose target aircraft has more than one task (`l_n_flight` > 1) score 0.634 top-3, against 0.763 for single-task aircraft.
- **List length:** accuracy falls as `seq_len` grows, from 0.82 at ≤ 20 tasks to about 0.57 at 50+ (binary reference).
- **Controllers and scenarios:** scenarios are flat at 0.646–0.654. Controllers range from 0.58 to 0.73 for the binary reference.
- **Importance** (`reports/importance.md`; gain and held-out permutation, stable across folds): `task_index` dominates, and shuffling it drops top-3 by 0.37. Next come `distance_to_sector`, `l_rank_absf1_all`, `time_to_sector`, `k_timeToExitPoint`, `k_distanceToExitPoint`, `x_task_age` and `l_rank_age_type`, with drops of 0.02–0.08 each.

## 7. Leakage and causality

- Validation labels were used only to score each candidate once and in the final refit. Test labels were never available. No leaderboard probing was done.
- Hyper-parameters, iteration counts, smoothing α and blend weights come from CV only; the blend weights were fixed in advance.
- Features exclude `sample_id`, `session`, `atco`, `timestamp`, `task_name` and `scenario`. The callsign is used only as a within-sample grouping key, in rejected groups. The temporal features and smoothing use only earlier snapshots of the same session. The rejected target encoding was strictly out-of-fold by controller.
- The final model uses only per-row features of the current snapshot, so it is causal by construction.

## 8. Limitations, unverified assumptions, next steps

- **Small holdouts.** Validation covers only 2 controllers and 3 sessions, so its CIs are wide. CV and validation disagree on lambdarank vs binary. The private leaderboard (controller 023, 948 samples) may favour either; the blend hedges.
- **Assumed round scaling.** Iteration counts for the full refits are scaled linearly with data size (R × N_full / N_fold_train). This is a heuristic, not validated.
- **Probable domain shift.** The rule baseline scores 0.485 on validation vs 0.352 on test and 0.356 on CV, which suggests different task mixes or annotation per controller.
- **Not done in time:** CatBoost (aborted for time), a per-type or two-stage model, and permutation importance beyond the top-20 gain features.
- **Next steps:**
  1. Targeted features for exit/entry coordination, the weakest large classes, built from coordination-state timing relative to sector entry and exit.
  2. 5-seed averages to reduce variance further.
  3. A nested check of the lambdarank/binary weight using validation as a fifth fold.
  4. Revisiting target encoding with stronger smoothing, since it helped on validation but not on CV.

## 9. Reproduce the final submission

```bash
PY=.venv/bin/python
$PY -m src.validate_data
$PY -m src.train cv lgb_binary_3s lgb_lambdarank_3s         # CV + train-only validation scores, iteration counts
$PY -m src.blend_fixed lgb_binary_3s lgb_lambdarank_3s      # equal-weight blend: CV + validation
$PY -m src.predict lgb_binary_3s,lgb_lambdarank_3s --val-from-cv
$PY scorer.py --labelled validation.csv submissions/validation_submission.csv   # 0.6589
$PY -m src.train final lgb_binary_3s                        # train+validation refit, R_final = 258, seeds 42–44
$PY -m src.train final lgb_lambdarank_3s                    # R_final = 185, seeds 42–44
$PY -m src.predict lgb_binary_3s,lgb_lambdarank_3s --out submissions/submission.csv
$PY -m src.validate_submission submissions/submission.csv
```

The earlier single-model submission, binary LightGBM with 3 seeds refit on train + validation, is kept as `submissions/submission_stage3_lgb_binary.csv`.
