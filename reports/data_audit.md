# Stage 0 data audit

**Failed checks: 3**: train: no negative ages/since columns (violations: {'k_levelClearanceAgeS': 158, 'k_headingClearanceAgeS': 161}); validation: no negative ages/since columns (violations: {'k_levelClearanceAgeS': 12, 'k_headingClearanceAgeS': 6}); test: no negative ages/since columns (violations: {'k_levelClearanceAgeS': 19, 'k_headingClearanceAgeS': 13})

## Columns

- OK: validation columns == train columns
- OK: test columns == train columns minus target_index
- OK: feature column count = 86 (expected 86)
- OK: all 10 identifier/label columns present
- prefix group sizes: x_8, k_32, a_12, l_9, other 25

## Counts vs documentation

| split | sessions | samples | rows | cols | controllers | sessions list |
|---|---:|---:|---:|---:|---|---|
| train | 34 | 11923 | 432578 | 96 | [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 17, 18] | 001sc1, 002sc2, 002sc3, 003sc1, 003sc3, 004sc1, 004sc2, 005sc1, 005sc2, 005sc3, 006sc2, 007sc1, 007sc3, 008sc1, 008sc2, 009sc1, 009sc3, 010sc3, 011sc2, 011sc3, 014sc1, 014sc2, 015sc1, 015sc2, 015sc3, 016sc1, 016sc2, 016sc3, 017sc1, 017sc2, 017sc3, 018sc1, 018sc2, 018sc3 |
| validation | 3 | 1252 | 42110 | 96 | [19, 20] | 019sc1, 019sc2, 020sc1 |
| test | 5 | 1863 | 66409 | 95 | [20, 23] | 020sc2, 020sc3, 023sc1, 023sc2, 023sc3 |

- OK: train sessions = 34 (documented 34)
- OK: train samples = 11923 (documented 11923)
- OK: train rows = 432578 (documented 432578)
- OK: train cols = 96 (documented 96)
- OK: validation sessions = 3 (documented 3)
- OK: validation samples = 1252 (documented 1252)
- OK: validation rows = 42110 (documented 42110)
- OK: validation cols = 96 (documented 96)
- OK: test sessions = 5 (documented 5)
- OK: test samples = 1863 (documented 1863)
- OK: test rows = 66409 (documented 66409)
- OK: test cols = 95 (documented 95)

## Per-sample invariants

- OK: train: rows per sample == seq_len
- OK: train: row_index is exactly 0..seq_len-1
- OK: train: seq_len/session/atco/scenario/timestamp constant within sample (violations: [])
- OK: train: task_name unique within sample
- train: seq_len range 9–59, mean 36.28
- OK: train: exactly one target_index==1 per sample (counts: {1: 11923})
- OK: train: target_index in {0,1}
- OK: validation: rows per sample == seq_len
- OK: validation: row_index is exactly 0..seq_len-1
- OK: validation: seq_len/session/atco/scenario/timestamp constant within sample (violations: [])
- OK: validation: task_name unique within sample
- validation: seq_len range 11–49, mean 33.63
- OK: validation: exactly one target_index==1 per sample (counts: {1: 1252})
- OK: validation: target_index in {0,1}
- OK: test: rows per sample == seq_len
- OK: test: row_index is exactly 0..seq_len-1
- OK: test: seq_len/session/atco/scenario/timestamp constant within sample (violations: [])
- OK: test: task_name unique within sample
- test: seq_len range 10–54, mean 35.65

## Task types

- train: task_index values [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13]
- OK: train: task_name prefix matches task_index (0 mismatches)
- OK: train: task type 11 absent
- validation: task_index values [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13]
- OK: validation: task_name prefix matches task_index (0 mismatches)
- OK: validation: task type 11 absent
- test: task_index values [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13]
- OK: test: task_name prefix matches task_index (0 mismatches)
- OK: test: task type 11 absent

| task_index | type | positives (train) | rows (train) | pos rate |
|---:|---|---:|---:|---:|
| 0 | entry_conflict_resolution | 31 | 8194 | 0.0038 |
| 1 | exit_conflict_resolution | 186 | 5558 | 0.0335 |
| 2 | entry_coordination | 940 | 128283 | 0.0073 |
| 3 | exit_coordination | 704 | 54845 | 0.0128 |
| 4 | assume_aircraft | 3003 | 71633 | 0.0419 |
| 5 | transfer_aircraft | 3145 | 27786 | 0.1132 |
| 6 | conflict_resolution | 351 | 10000 | 0.0351 |
| 7 | non_conformance_resolution | 86 | 1297 | 0.0663 |
| 8 | qos_improvement | 237 | 27461 | 0.0086 |
| 9 | entry_conditions | 2 | 125 | 0.0160 |
| 10 | exit_conditions | 407 | 33400 | 0.0122 |
| 12 | zone_conflict | 1950 | 48719 | 0.0400 |
| 13 | return_to_route | 881 | 15277 | 0.0577 |

## Aircraft-level columns (within sample, same subject callsign)

- OK: subject callsign parsed for every train row
- groups (sample, aircraft) with >1 task: 128965
- columns that vary between tasks of the same aircraft (share of such groups):
  - `x_task_age`: 0.948
  - `l_frac_type`: 0.944
  - `l_n_type`: 0.944
  - `l_rank_age_type`: 0.746
  - `l_rank_f0_type`: 0.595
  - `l_rank_f1_type`: 0.594
  - `l_rank_absf1_type`: 0.556
- assume/entry_coordination twins (67090 pairs): varying columns `x_task_age` 0.941, `l_n_type` 0.933, `l_frac_type` 0.933, `l_rank_age_type` 0.682, `l_rank_f1_type` 0.205, `l_rank_f0_type` 0.204, `l_rank_absf1_type` 0.197
- train targets with l_twin_AE=1: 3535 / 11923; by type {4: 2600, 2: 924, 0: 11}

## Missingness (fraction NaN)

| column | train | validation | test | train is_assumed=1 | train is_assumed=0 |
|---|---:|---:|---:|---:|---:|
| `distance_to_MD` | 0.862 | 0.878 | 0.859 | 0.836 | 0.885 |
| `time_to_MD` | 0.862 | 0.878 | 0.859 | 0.836 | 0.885 |
| `exit_FL` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `distance_to_zone` | 0.569 | 0.600 | 0.591 | 0.654 | 0.496 |
| `k_isClimbing` | 0.539 | 0.566 | 0.496 | 0.000 | 1.000 |
| `k_isDescending` | 0.539 | 0.566 | 0.496 | 0.000 | 1.000 |
| `k_isAtCFL` | 0.601 | 0.641 | 0.603 | 0.136 | 1.000 |
| `k_isClimbingToCFL` | 0.601 | 0.641 | 0.603 | 0.136 | 1.000 |
| `k_isDescendingToCFL` | 0.601 | 0.641 | 0.603 | 0.136 | 1.000 |
| `k_clearedFLDelta` | 0.601 | 0.641 | 0.603 | 0.136 | 1.000 |
| `k_levelClearanceAgeS` | 0.601 | 0.641 | 0.603 | 0.136 | 1.000 |
| `k_headingClearanceAgeS` | 0.879 | 0.931 | 0.878 | 0.738 | 1.000 |
| `k_timeToExitPoint` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `k_distanceToExitPoint` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `k_exitConditionsAreMet` | 0.657 | 0.699 | 0.673 | 0.255 | 1.000 |
| `k_aircraftNeedsToBeTransferred` | 0.683 | 0.683 | 0.661 | 0.312 | 1.000 |
| `k_isAtExitFL` | 0.539 | 0.566 | 0.496 | 0.000 | 1.000 |
| `k_hasToClimbToExitFL` | 0.539 | 0.566 | 0.496 | 0.000 | 1.000 |
| `k_hasToDescendToExitFL` | 0.539 | 0.566 | 0.496 | 0.000 | 1.000 |
| `k_isClearedHeadingAssigned` | 0.539 | 0.566 | 0.496 | 0.000 | 1.000 |
| `k_outCoordMismatch` | 0.456 | 0.514 | 0.434 | 0.189 | 0.684 |
| `k_outCoordRevision` | 0.455 | 0.514 | 0.434 | 0.187 | 0.684 |
| `k_isPassingCFL` | 0.654 | 0.689 | 0.670 | 0.250 | 1.000 |
| `k_isFlightLevelBust` | 0.601 | 0.641 | 0.603 | 0.136 | 1.000 |
| `k_isMaintainingFL` | 0.539 | 0.566 | 0.496 | 0.000 | 1.000 |
| `k_clearedFL` | 0.375 | 0.388 | 0.423 | 0.136 | 0.580 |
| `k_cfl_minus_fl` | 0.375 | 0.388 | 0.423 | 0.136 | 0.580 |
- OK: train: is_assumed has no NaN
- OK: validation: is_assumed has no NaN
- OK: test: is_assumed has no NaN

## Value checks

- **FAIL**: train: no negative ages/since columns (violations: {'k_levelClearanceAgeS': 158, 'k_headingClearanceAgeS': 161})
- train: rows with |horizontal route deviation| > 100 km: 365
- train: 51 columns are binary {0,1} (ignoring NaN)
- train: fully duplicated rows (ignoring sample_id): 0
- train: time_to_sector range [-439, 2148]
- **FAIL**: validation: no negative ages/since columns (violations: {'k_levelClearanceAgeS': 12, 'k_headingClearanceAgeS': 6})
- validation: rows with |horizontal route deviation| > 100 km: 0
- validation: 52 columns are binary {0,1} (ignoring NaN)
- validation: fully duplicated rows (ignoring sample_id): 0
- validation: time_to_sector range [-369, 2022]
- **FAIL**: test: no negative ages/since columns (violations: {'k_levelClearanceAgeS': 19, 'k_headingClearanceAgeS': 13})
- test: rows with |horizontal route deviation| > 100 km: 36
- test: 51 columns are binary {0,1} (ignoring NaN)
- test: fully duplicated rows (ignoring sample_id): 0
- test: time_to_sector range [-434, 2057]

## Timestamps and sessions

- OK: train: timestamps strictly increasing with sample_id within each session (non-increasing steps: 0, midnight wraps: 0)
- train: gap between consecutive samples: =3s 0.676, median 3s, max 281s
- train: sample_ids contiguous per session: True
- OK: validation: timestamps strictly increasing with sample_id within each session (non-increasing steps: 0, midnight wraps: 0)
- validation: gap between consecutive samples: =3s 0.714, median 3s, max 191s
- validation: sample_ids contiguous per session: True
- OK: test: timestamps strictly increasing with sample_id within each session (non-increasing steps: 0, midnight wraps: 0)
- test: gap between consecutive samples: =3s 0.706, median 3s, max 181s
- test: sample_ids contiguous per session: True
- OK: sessions disjoint across files
- controller overlap train∩val [], train∩test [], val∩test [np.int16(20)]

## Label persistence (descriptive)

- train: consecutive same target task_name 0.721; previous target still in list 0.801
- train: mean relative target position row_index/(seq_len-1) = 0.516
- validation: consecutive same target task_name 0.765; previous target still in list 0.821
- validation: mean relative target position row_index/(seq_len-1) = 0.525

## sample_submission.csv

- OK: sample_submission header
- OK: sample_submission ids == test ids (1863 rows, 1863 test samples)
- first row: `12797,9 6 41` (docs show `12797,9 6 4`)

