# P5 — Brainstorming di feature di dominio (non ancora implementate)

Base: feature set v5 (`src/features.py`). Le evidenze vengono dall'EDA su train+validation del 2026-10-20 (numeri sotto).

## Evidenze dall'EDA

| Fatto | Valore |
|---|---|
| Target uguale al target dello snapshot precedente (stessa sessione) | 72% |
| … se lo snapshot precedente è a ≤3.5 s | **95%** |
| … se lo snapshot precedente è a >3.5 s | 25% |
| Quota di snapshot a ≤3.5 s dal precedente: train+val / test | 68% / 70% |
| Task presenti anche nello snapshot precedente (gap ≤3.5 s) | 99% |
| `x_state_age` resettato rispetto allo snapshot precedente: target / non target | 1.2% / 0.2% |
| L'ATCO ha agito sull'aereo dallo snapshot precedente: target / non target | 4.1% / 0.5% |
| Target su un aereo con >1 task aperto | 86% (ed è il task più giovane dell'aereo solo nel 52%) |
| Target per tipo: transfer 27%, assume 25%, zone_conflict 16%, entry_coord 8%, return_to_route 7% | |

Il test contiene sessioni complete con timestamp monotoni, quindi le feature temporali sono calcolabili anche lì.

## 10 proposte, ordinate per guadagno atteso / costo

| # | Nome | Formula | Colonne | Ipotesi | Rischio leakage | Costo |
|---|---|---|---|---|---|---|
| 1 | ~~Smoothing temporale degli score~~ (post-processing, **scartato**, vedi sotto) | `score'_t = score_t + w·score_{t-1}` per lo stesso `task_name` se gap ≤ 3.5 s (eventualmente ricorsivo, decadimento esponenziale) | score OOF, `session`, `timestamp`, `task_name` | un task dura 5–40 s, quindi snapshot adiacenti hanno lo stesso target nel 95% dei casi | **nessuno** se usa solo snapshot passati e score (non le label); `w` va scelto con la CV | basso |
| 2 | **Lag delle feature di attività** | `Δa_since_flight`, `Δx_state_age`, `Δa_n60` rispetto allo stesso `task_name` nello snapshot precedente; flag `state_reset`, `acted_since_prev` | `a_*`, `x_state_age`, `session`, `timestamp` | un'azione dell'ATCO o un cambio di stato del task tra due snapshot segnala il task in lavorazione (4.1% vs 0.5%) | nessuno (solo passato) | medio |
| 3 | `gap_prev_s` e `is_new_task` | secondi dallo snapshot precedente; task assente nello snapshot precedente | `timestamp`, `task_name` | dopo un gap lungo il controllore cambia task (la persistenza cala al 25%); un task appena comparso attira attenzione | nessuno | basso |
| 4 | Task chiusi dall'ultimo snapshot | n. task spariti dello stesso aereo / dello snapshot | `task_name` lag | se il task precedente si chiude, il prossimo spesso è il fratello sullo stesso aereo (es. exit_coord → transfer) | nessuno | medio |
| 5 | Urgenza di uscita | `k_timeToExitPoint` vs soglia di transfer; `k_aircraftNeedsToBeTransferred·(1/k_timeToExitPoint)`; flag `exit_conditions_met & !is_transferred` | `k_*`, `is_transferred` | il transfer (27% dei target) ha una finestra temporale precisa prima del confine | basso | basso |
| 6 | Urgenza di ingresso | `sec_to_entry` vs `x_task_age` di assume/entry; flag `inside & !assumed` | `time_to_sector`, `is_assumed`, `x_task_age` | assume (25% dei target) parte quando l'aereo chiama, intorno al confine | basso | basso |
| 7 | Priorità del tipo condizionata alla coppia (target encoding OOF) | P(selezione del tipo | insieme dei tipi sull'aereo, fascia di `abs_tts`), calcolata **dentro il fold** | `task_index`, `ac_has_*`, `abs_tts` | regola del 3:1 assume vs entry_coord, documentata nell'analisi | **alto** se calcolata fuori dal fold |
| 8 | Severità del conflitto | `1/time_to_MD`, `distance_to_MD` rank tra i conflitti, `multiple_conflicts_ind·HMI_alert` | `*_MD`, `conflict_ind` | zone_conflict e conflict_res pesano il 21% dei target; urgenza ∝ tempo alla minima distanza | basso | basso |
| 9 | Coerenza di livello | `|current_FL − exit_FL|`, `k_hasToClimb/Descend × k_timeToExitPoint` | `current_FL`, `exit_FL`, `k_*` | serve una clearance di livello prima dell'uscita | basso | basso |
| 10 | Stile del controllore (non supervisionato) | statistiche della sessione corrente fino a t: frequenza di azioni (`a_n300` medio), ritmo dei transfer | `a_*` cumulati per sessione | adatta il modello al ritmo di un ATCO mai visto (023) senza usare le label | nessuno (solo passato e solo feature) | medio |

## Verifica rapida della #1 (già fatta, sugli OOF di m2-v0)

| finestra | w=0 | 0.3 | 0.5 | 0.7 | 0.85 |
|---|---|---|---|---|---|
| gap ≤ 3.5 s | 0.6630 | 0.6613 | 0.6601 | 0.6597 | 0.6588 |
| gap ≤ 6.5 s | 0.6630 | 0.6610 | 0.6597 | 0.6594 | 0.6587 |

**Scartata**: peggiora sempre e migliora al massimo 5 ATCO su 18. Le feature di due snapshot adiacenti sono quasi identiche, quindi le predizioni del modello sono già persistenti. La persistenza al 95% riguarda le label, che al test non ci sono, e mediare gli score non aggiunge informazione. Il segnale temporale utile è quello che *cambia* tra snapshot (#2, #3, #4), non quello che resta uguale.

## Le 3 da fare per prime

1. **Lag delle feature di attività (#2) + `gap_prev_s`/`is_new_task` (#3)**: catturano i cambiamenti tra snapshot (azioni dell'ATCO, reset di stato, task nuovi), che sono 8 volte più frequenti sul target.
2. **Task chiusi dall'ultimo snapshot (#4)**: quando il task precedente sparisce, il prossimo target spesso è il fratello sullo stesso aereo.
3. **Urgenza di uscita/ingresso (#5, #6)**: riguardano transfer e assume, che insieme sono il 52% dei target. Costo bassissimo.

Nota di metodo: tutte le proposte usano solo snapshot **passati** della stessa sessione. Usare anche snapshot futuri del test sarebbe tecnicamente possibile offline, ma non è realizzabile in un sistema operativo reale. Andrebbe deciso col team e spiegato alla giuria, quindi lo lascio fuori.
