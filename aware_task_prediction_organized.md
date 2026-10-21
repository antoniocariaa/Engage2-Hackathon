# AWARE — Predicting the Controller’s Active Task

## 1. Overview

An en-route air traffic controller looks after a block of airspace and keeps a mental list of jobs: accepting aircraft from the previous sector, keeping aircraft apart, steering them around military zones, and handing them over to the next sector.

The AWARE research system builds that list automatically from radar, flight plan, and coordination data. Every 3 seconds, it holds a list of dozens of open tasks.

The goal is to predict which task in the list the controller is working on at that moment. A model that can do this could help a future controller support system decide what to show first.

## 2. Goal

For every sample in the test set, submit the three tasks most likely to be the one the controller is working on, ordered from most likely to least likely.

A sample scores one point when the true task is among the three predictions.

## 3. Competition timeline

- **Start:** 2 days ago
- **Close:** 3 hours to go

## 4. Data source and sample structure

### 4.1 Data source

The data was recorded in simulator exercises at the Swedish upper-airspace sector **ESMM_23**. Nineteen controllers each flew up to three traffic scenarios.

AWARE ran on the recorded traffic and wrote out its task list every 3 seconds. Experts then marked, from the recordings, which task the controller was working on at each moment.

The dataset comes from en-route air traffic control simulator exercises recorded in September 2025. There are 19 controllers, up to three scenarios per controller, and 42 sessions in total.

### 4.2 What a sample contains

- A **sample** is one snapshot of the task list during a labelled moment. It is identified by `sample_id`.
- A **row** is one task in that list.
- `row_index` gives a task’s position in the list, starting at `0`.
- `seq_len` gives the number of tasks in the list.
- Exactly one task per labelled sample is the answer. It is marked `target_index = 1` in the training and validation files.
- Each task has a type (`task_index`, one of 14 types) and belongs to an aircraft, for example `SAS95T`.
- `task_name` combines the task type and aircraft callsign, for example `assume_aircraft_SAS95T`.
- The 86 feature columns describe aircraft position and status, conflicts, coordination with neighbouring sectors, recent controller inputs, and the task’s position in the list. Every column is explained in `FEATURES.md`.

### 4.3 Important properties of the data

- **Aircraft vs. task:** Most columns describe the aircraft and are identical for all tasks belonging to the same aircraft in a sample. Distinguishing an aircraft’s tasks relies on the task type, task age (`x_task_age`), and list-context columns (`l_`).
- **Difficult task pair:** The hardest case is an aircraft with both an `assume_aircraft` task and an `entry_coordination` task.
- **Empty cells:** An empty cell means “does not apply” or “not computed”; it never means `0`. Many `k_` columns are empty for aircraft the controller does not yet control.
- **Related samples:** Snapshots are 3 seconds apart, and consecutive samples from one session usually share the same answer.
- **Session-based split:** The split is by whole sessions, so test controllers and sessions are not in the training data. Any custom validation split should also keep sessions separate. A random split over samples gives far too optimistic scores.
- **Repeated traffic:** The three scenarios repeat across sessions with different controllers. Memorising one controller’s choices for a scenario will not perform well.
- **Imperfect labels:** The labels come from human annotation and contain some noise.
- **Human factors:** Controllers use different strategies and may give different priorities to tasks in very similar traffic situations.

## 5. Dataset files

| File | Contents | Samples | Rows |
|---|---|---:|---:|
| `train.csv` | Training data with `target_index` | 11,923 | 432,578 |
| `validation.csv` | Labelled data from sessions `019sc1`, `019sc2`, and `020sc1` | 1,252 | 42,110 |
| `test.csv` | Data to predict, without `target_index` | 1,863 | 66,409 |
| `sample_submission.csv` | Valid submission example (simple baseline) | 1,863 | — |
| `FEATURES.md` | Description of every column | — | — |
| `scorer.py` | Script for checking submissions offline | — | — |

### 5.1 Training data

`train.csv` contains 34 sessions, 11,923 samples, 432,578 rows, and 96 columns. The label column `target_index` is `1` for the task the controller was working on and `0` for every other task in that sample. Exactly one row per `sample_id` is labelled `1`.

### 5.2 Validation data

`validation.csv` contains three further sessions (`019sc1`, `019sc2`, and `020sc1`), with 1,252 samples, 42,110 rows, and the same 96 columns as the training data.

Use it to evaluate the model on sessions it has not seen. The validation data can also be merged with the training data for a final model. If creating a custom split, keep whole sessions together.

### 5.3 Test data

`test.csv` contains five sessions from controllers 020 and 023:

- `020sc2`
- `020sc3`
- `023sc1`
- `023sc2`
- `023sc3`

It contains 1,863 samples and 66,409 rows. It has the same columns as `train.csv`, except that `target_index` is hidden.

For each `sample_id`, submit three `row_index` values. Controller 020 also flew one validation session; the two test controllers, 020 and 023, do not appear in `train.csv`.

### 5.4 Baseline submission

`sample_submission.csv` is a template for a valid submission. Its predictions use a simple rule: task types that are most often the answer in `train.csv` are ranked first, and within a type the aircraft closest to the sector boundary is selected.

The stated test-set top-3 hit rate for this baseline is **0.352**.

### 5.5 Dataset totals

The dataset contains:

- **15,038 samples**
- **541,097 rows**
- **9 to 59 tasks per sample**
- **36 tasks per sample on average**
- **96 columns** in the labelled training and validation files

The data is split by whole sessions: each session appears in exactly one file. The public and private leaderboard subsets are described in the evaluation section.

## 6. Evaluation

### 6.1 Metric: top-3 hit rate

For each test sample, the first three distinct task IDs in the prediction are considered. A sample scores `1` if the labelled task is among those three, otherwise it scores `0`.

The final score is the mean across all 1,863 test samples, with every sample counting equally.

\[
\\text{Top-3 hit rate} = \\frac{1}{N}\\sum_{s=1}^{N}\\text{hit}(s)
\]

where `N` is the number of test samples (1,863), and `hit(s)` is `1` if the labelled task for sample `s` is among its first three predicted tasks, and `0` otherwise.

### 6.2 Leaderboards

- **Public leaderboard:** samples from controller 020, sessions `020sc2` and `020sc3` — 915 samples.
- **Private leaderboard:** samples from controller 023, sessions `023sc1`, `023sc2`, and `023sc3` — 948 samples. This leaderboard decides the final ranking.

### 6.3 Reference scores

| Method | Top-3 hit rate |
|---|---:|
| Random choice of three tasks | 0.094 |
| `sample_submission.csv`: most frequently labelled task types first, then aircraft closest to sector boundary | 0.352 |

### 6.4 Offline validation

To check a submission on the validation set, run:

```bash
python scorer.py --labelled validation.csv my_validation_submission.csv
```

The scoring script also reports invalid submissions, including missing or duplicated `sample_id` values and non-numeric IDs.

## 7. Submission format

The submission must be a CSV with a header and one line per test sample.

```csv
sample_id,prediction
12797,9 6 4
12798,9 6 4
12799,10 7 4
```

### Requirements

- `sample_id`: every `sample_id` in `test.csv`, included exactly once.
- `prediction`: the `row_index` values of the chosen tasks, separated by spaces and ordered from best to worst.
- Only the first three distinct IDs count. More IDs are allowed but ignored.
- Repeated IDs count only once.
- Fewer than three IDs are allowed, but can lower the score.
- A missing `sample_id`, a duplicated `sample_id`, or a non-numeric ID makes the submission invalid.

## 8. Dataset features

### 8.1 Background

Air traffic controllers (ATCOs) each look after one block of airspace, called a **sector**. The sector in this dataset is `ESMM_23`, part of the Swedish upper airspace. Aircraft enter the sector from a neighbouring sector, cross it, and leave into the next sector.

Typical controller tasks include:

- accepting an aircraft from the previous sector;
- giving clearances, such as instructions on level, heading, or route;
- keeping aircraft separated;
- handing an aircraft over to the next sector.

AWARE monitors the traffic and maintains a list of these jobs, called **tasks**.

A sample is a snapshot taken every 3 seconds while the controller is working on a labelled task. Most columns describe the aircraft, not the task. The task type, task age, and some list-context columns distinguish tasks belonging to the same aircraft.

**Units:** distances are in metres unless stated otherwise; `NM` means nautical miles. Times are in seconds. Flight levels are in hundreds of feet (`FL 350 = 35,000 ft`).

### 8.2 Identifiers and labels

| Column | Definition |
|---|---|
| `sample_id` | Identifier of a sample (one snapshot). All rows with the same `sample_id` belong to one task list, and exactly one is the answer. |
| `seq_len` | Number of tasks in the sample’s list (9 to 59). |
| `row_index` | Position of the task in the list, starting at `0`. Submissions identify tasks using this value. |
| `target_index` | Label, present only in train and validation: `1` for the task the controller was working on and `0` for all others. Exactly one row per sample has `1`. |
| `task_index` | Task type code. It is a category, not a quantity: code `12` is not “more” than code `4`. |
| `session` | Exercise run, written as controller and scenario, e.g. `004sc2` is controller 004 flying scenario 2. Training, validation, and test use different sessions. |
| `atco` | Controller number (1 to 23; 19 controllers in total). |
| `scenario` | Traffic scenario (1, 2, or 3). Sessions of the same scenario use the same traffic, flown by different controllers. |
| `timestamp` | Snapshot time (`HH:MM:SS`, UTC, real-world clock; the date is removed). Snapshots are 3 seconds apart. Consecutive samples from one session often share the same answer. |
| `task_name` | Task type followed by the aircraft callsign, e.g. `assume_aircraft_SAS95T`. Conflict tasks name two aircraft, e.g. `conflict_resolution_DLH3FR_FIN953`; the first aircraft named is the aircraft the task is about. A callsign is the aircraft’s radio name, usually airline code plus flight number. |

### 8.3 Task types (`task_index`)

| Code | Task type | Description |
|---:|---|---|
| 0 | `entry_conflict_resolution` | Resolve a predicted conflict between two aircraft near the point where they enter the sector. |
| 1 | `exit_conflict_resolution` | Resolve a predicted conflict near the point where they leave the sector. |
| 2 | `entry_coordination` | Agree with the previous sector on the conditions (level, time) at which the aircraft will enter. |
| 3 | `exit_coordination` | Agree with the next sector on the conditions at which the aircraft will leave. |
| 4 | `assume_aircraft` | Take control of an aircraft offered by the previous sector. |
| 5 | `transfer_aircraft` | Hand the aircraft over to the next sector. |
| 6 | `conflict_resolution` | Resolve a predicted conflict between two aircraft inside the sector. |
| 7 | `non_conformance_resolution` | Deal with an aircraft that is not following its clearance. |
| 8 | `qos_improvement` | Offer the aircraft something better, such as a shorter direct route. |
| 9 | `entry_conditions` | Adjust the conditions at which the aircraft enters (very rare). |
| 10 | `exit_conditions` | Bring the aircraft to the level agreed for leaving the sector. |
| 11 | `aircraft_request` | Answer a pilot’s request (does not occur in this data). |
| 12 | `zone_conflict` | The aircraft’s route crosses an active military zone that it must avoid. |
| 13 | `return_to_route` | The aircraft is off its route and has to be cleared back onto it. |

### 8.4 Aircraft position and status

| Column | Definition |
|---|---|
| `distance_to_sector` | Horizontal distance from the aircraft to the sector boundary, in metres. Positive means outside the sector and indicates how far it still has to fly to enter; negative means inside and indicates the distance to the boundary it will leave by. |
| `time_to_sector` | `distance_to_sector` divided by aircraft speed: time in seconds until the aircraft reaches the boundary. Negative inside the sector. Capped at 3,600 seconds. |
| `within_sector` | `1` if the aircraft is inside the sector horizontally. |
| `approaching_sector` | `1` if the aircraft is outside the sector and getting closer to it. |
| `current_FL` | Current aircraft flight level (FL 100 to 400). Aircraft below FL 100, e.g. departures still climbing out, have been removed. |
| `exit_FL` | Flight level at which the aircraft is planned to leave the sector. Empty when unknown. |
| `is_assumed` | `1` if this controller is in control of the aircraft. |
| `is_not_assumed` | `1` if this controller is not (yet) in control of the aircraft, typically because it is still with the previous sector. |
| `is_transferred` | `1` if control has already been handed over to the next sector (rare in this data). |

### 8.5 Conflicts between aircraft

| Column | Definition |
|---|---|
| `conflict_ind` | `1` if AWARE predicts a conflict within the next 10 minutes: this aircraft and another one will come closer than the minimum allowed separation. |
| `distance_to_MD` | Distance in nautical miles the aircraft will fly along its predicted path until the moment of minimum distance to the other aircraft. Empty when there is no conflict. |
| `time_to_MD` | Time in seconds until the moment of minimum distance (`0` to `600`). Empty when there is no conflict. A value of `0` means the aircraft are closest now. |
| `multiple_conflicts_ind` | `1` if the aircraft has more than one predicted conflict. |
| `HMI_conflict_safety_alert` | `1` if the controller’s screen shows a short-term conflict alert (STCA) for this aircraft. This is the ground system’s last-minute warning that two aircraft are about to get too close. |
| `CPA_entry_location` | `1` if the closest point of approach of the conflict lies where the aircraft enters the sector. |
| `CPA_exit_location` | `1` if the closest point of approach lies where the aircraft leaves the sector. |
| `what_if_potential_conflict` | `1` if a “what-if” check finds that a planned change, such as a new level, would create a conflict (rare). |

### 8.6 Route, clearances, and conformance

| Column | Definition |
|---|---|
| `qos_candidate` | `1` if the aircraft could be offered a better service (QoS), such as a direct route that saves time and fuel. |
| `monitoring_rule_activated` | `1` if one of AWARE’s conformance checks is active for the aircraft, or it has drifted off its route. Conformance means doing what the clearance says. |
| `non_conformance_FL_delta` | Number of flight levels the aircraft is away from its cleared level while it is not conforming; `0` otherwise. |
| `non_conformance_heading_delta` | Number of degrees the aircraft’s heading differs from its cleared heading while it is not conforming; `0` otherwise. |
| `non_conformance_agreed_route_deviation_fl` | Vertical difference, in flight levels, between the aircraft and the level profile of its agreed route. `0` means on profile. |
| `non_conformance_agreed_route_deviation_horizontally` | Sideways distance in metres between the aircraft and its agreed route (cross-track distance). `0` means on the route. A few very large values (over 100 km) may be data errors. |

### 8.7 Military zones

| Column | Definition |
|---|---|
| `zone_conflict_ind` | `1` if the aircraft’s route crosses a temporary military zone (TRA) that is currently active. Civil traffic must be kept out of such a zone. |
| `distance_to_zone` | Distance in nautical miles to the closest military zone that the route crosses. Empty when the route crosses none. |

### 8.8 Hand-over from the previous sector (`x_` columns)

Before an aircraft enters, the previous sector and this sector agree on the entry conditions (coordination). The previous sector then offers the aircraft (transfer), and this controller accepts it (assume).

| Column | Definition |
|---|---|
| `x_isc_notified` | `1` if coordination with the previous sector is at the “notified” stage: the flight has been announced but the entry conditions are not yet agreed. |
| `x_isc_coordinated` | `1` if entry conditions have been agreed. If both `x_isc_` columns are `0`, coordination has not started. |
| `x_transfer_in` | `1` if the previous sector has offered the aircraft to this controller and is waiting for it to be accepted. |
| `x_request_in` | `1` if this controller has asked to take the aircraft early (rare). |
| `x_entry_coord_not_done` | `1` if AWARE considers the entry coordination still to be done. |
| `x_in_sector_h` | `1` if the aircraft is inside the sector horizontally (AWARE’s own test, close to `within_sector`). |
| `x_state_age` | Seconds since the aircraft’s control status last changed, for example since it was offered by the previous sector. Capped at 3,600. |
| `x_task_age` | Seconds since this task appeared in AWARE’s task list. Short interruptions (up to 30 seconds) are ignored. This is one of the few columns that differ between tasks of the same aircraft. |

### 8.9 Aircraft state and exit (`k_` columns)

These values come from AWARE’s internal reasoning about each aircraft. Many are only computed for aircraft the controller is in control of; elsewhere they are empty.

| Column | Definition |
|---|---|
| `k_isClimbing` | `1` if the aircraft is climbing. |
| `k_isDescending` | `1` if the aircraft is descending. |
| `k_isMaintainingFL` | `1` if the aircraft is flying level. |
| `k_isAtCFL` | `1` if the aircraft is at its cleared flight level (CFL), the level the controller has told it to fly. |
| `k_isClimbingToCFL` | `1` if the aircraft is climbing towards its cleared level. |
| `k_isDescendingToCFL` | `1` if the aircraft is descending towards its cleared level. |
| `k_isPassingCFL` | `1` if the aircraft is just passing through its cleared level. |
| `k_isFlightLevelBust` | `1` if the aircraft has left its cleared level without permission (a “level bust”). |
| `k_clearedFL` | The cleared flight level. |
| `k_cfl_minus_fl` | Cleared level minus current level, in flight levels. Positive means the aircraft still has to climb; negative means it has to descend. |
| `k_clearedFLDelta` | Current level minus cleared level (the opposite sign), computed only for aircraft under this controller’s control. |
| `k_isClearedFLAssigned` | `1` if a cleared level exists for the aircraft. |
| `k_levelClearanceAgeS` | Seconds since the last level instruction was given. |
| `k_isClearedHeadingAssigned` | `1` if the aircraft has been given a heading, a compass direction to fly instead of following its route. |
| `k_headingClearanceAgeS` | Seconds since the heading instruction. |
| `k_isLocked` | `1` if the aircraft has a heading or climb/descent rate instruction, so it is not simply following its planned route. |
| `k_groundSpeed` | Speed over the ground, in metres per second. |
| `k_trackAgeS` | Seconds since the last radar update for the aircraft (almost always `0`). |
| `k_inSectorVertically` | `1` if the aircraft is within the sector’s height band. |
| `k_distanceToExitPoint` | Distance in metres to the point where the aircraft is planned to leave the sector. |
| `k_timeToExitPoint` | Time in seconds to that exit point. |
| `k_isAtExitFL` | `1` if the aircraft is already at the level agreed for leaving the sector. |
| `k_hasToClimbToExitFL` | `1` if it still has to climb to reach that exit level. |
| `k_hasToDescendToExitFL` | `1` if it still has to descend to reach that exit level. |
| `k_exitConditionsAreMet` | `1` if the aircraft will leave the sector as agreed with the next sector. |
| `k_aircraftNeedsToBeTransferred` | `1` if the aircraft is under this controller’s control, at its exit level, and close to the exit, so it is due to be handed over. |
| `k_out_NOTIFIED` | `1` if coordination with the next sector is at the “notified” stage. |
| `k_out_COORDINATED` | `1` if exit conditions have been agreed with the next sector. |
| `k_outCoordMismatch` | `1` if the agreed exit level or time no longer matches what the aircraft is cleared to or predicted to do, so the agreement may need to be revised. |
| `k_outCoordRevision` | Number of the latest coordination message exchanged with the next sector. Higher means more back-and-forth. |
| `k_coordinationToolActivation` | `1` if AWARE would open its coordination tool for the aircraft, e.g. because the exit level or route changed or the aircraft is close to the border. |
| `k_transferToolActivation` | `1` if AWARE would open its transfer (hand-over) tool for the aircraft. |

### 8.10 Controller inputs (`a_` columns)

Every instruction typed into the system is logged, including inputs from neighbouring sectors such as an offer of an aircraft. These columns summarise what was entered before the snapshot.

| Column | Definition |
|---|---|
| `a_since_flight` | Seconds since the last input concerning this aircraft. `3,600` means none in the last hour. |
| `a_last_assume_control` | `1` if the last input was taking control of the aircraft. |
| `a_last_force_assume_control` | `1` if the last input was taking control without waiting for the offer (forced assume). |
| `a_last_assign_level_clearance` | `1` if the last input was a level instruction. |
| `a_last_assign_lateral_clearance` | `1` if the last input was a heading, direct-to, or route instruction. |
| `a_last_initiate_transfer_of_control` | `1` if the last input was an offer to hand the aircraft over. |
| `a_last_decontrol` | `1` if the last input released control of the aircraft (very rare here). |
| `a_since_any` | Seconds since the last input on any aircraft in the session. |
| `a_n60` | Number of inputs on all aircraft in the last 60 seconds, a measure of how busy the moment is. |
| `a_n300` | Number of inputs on all aircraft in the last 5 minutes. |
| `a_is_last_cs` | `1` if this aircraft received the most recent input of the session. |
| `a_is_last2_cs` | `1` if this aircraft received one of the two most recent inputs. |

### 8.11 Task-list context (`l_` columns)

These columns compare a task with the other tasks in the same sample. Ranks start at `1`, and equal values share the lowest rank.

| Column | Definition |
|---|---|
| `l_n_type` | Number of tasks of the same type in the list. |
| `l_frac_type` | Share of the list that has this task’s type (0 to 1). |
| `l_n_flight` | Number of tasks belonging to the same aircraft in the list. |
| `l_twin_AE` | `1` if the aircraft has both an `assume_aircraft` and an `entry_coordination` task in the list. Such pairs are hard to distinguish because the aircraft columns are identical. |
| `l_rank_f1_type` | Rank by `time_to_sector` (with sign) among tasks of the same type. `1` means the smallest value, i.e. the aircraft deepest inside the sector. |
| `l_rank_absf1_type` | Rank by absolute time to the boundary among tasks of the same type. `1` means the aircraft closest to the boundary. |
| `l_rank_f0_type` | Rank by `distance_to_sector` (with sign) among tasks of the same type. |
| `l_rank_absf1_all` | Rank by absolute `time_to_sector` among all tasks in the list. |
| `l_rank_age_type` | Rank by task age among tasks of the same type. `1` means the task that has been waiting longest. |

## 9. Acronyms

| Acronym | Meaning |
|---|---|
| ATC | Air Traffic Control |
| ATCO | Air Traffic Controller (Officer) |
| ATM | Air Traffic Management |
| AWARE | The research project and system that produced the task lists (artificial situational awareness for controllers) |
| CFL | Cleared Flight Level: the level a controller has told an aircraft to fly |
| CPA | Closest Point of Approach: where two aircraft are predicted to be nearest to each other |
| ESMM | Malmö area control (ICAO code of the Swedish upper-airspace control centre); `ESMM_23` is the sector in this data |
| FL | Flight Level: altitude in hundreds of feet at standard pressure (`FL 350` ≈ 35,000 ft) |
| HMI | Human-Machine Interface: the controller’s screen and tools |
| ISC | Inter-Sector Coordination: agreement between neighbouring sectors on how an aircraft is handed over |
| MD | Minimum Distance: the smallest predicted distance between two aircraft in a conflict |
| NM | Nautical Mile (1,852 m) |
| QoS | Quality of Service: offering aircraft better routes or levels |
| STCA | Short-Term Conflict Alert |
| TRA | Temporary Reserved Area: airspace reserved for military use at certain times |
| UTC | Coordinated Universal Time |

## 10. License and file summary

- **License:** Attribution-NonCommercial-ShareAlike 3.0 IGO (CC BY-NC-SA 3.0 IGO)
- **Files listed:** 5 files
- **Total size:** 148.75 MB
