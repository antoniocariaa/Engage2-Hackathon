# CLAUDE.md — AWARE Task-Prioritisation (Engage #2)

Progetto: ranking dei task ATC aperti per snapshot; metrica **top-3 hit rate** (`scorer.py`). Baseline 0.35.
Piano: `PIANO_TECNICO.md`. Workflow agentico: `GUIDA_AGENTICA.md`. Log esperimenti: `experiments/EXPERIMENTS.md`.

## Vincoli
- Solo CPU (8 core, ~15 GB RAM). Usa `float32`, parquet, `category`. Niente modelli pesanti.
- Python in `.venv`. Dati in `data/` (gitignored). Non committare CSV/parquet/OOF/submission.
- Lingua dei documenti e dei commenti di progetto: italiano; identificatori nel codice in inglese.

## Regole non negoziabili
1. Validazione = `GroupKFold` per `atco` su train+validation. Mai split per riga o per `sample_id`.
2. La Public LB (ATCO 020) è solo un controllo di integrità; non usarla per scegliere feature o iperparametri.
3. Nessuna feature può dipendere da `target_index`. Funzioni di `src/features.py` pure e senza stato.
4. Non modificare `scorer.py`, `train.csv`, `validation.csv`, `test.csv`, `sample_submission.csv`.
5. Righe ordinate per `sample_id` contiguo prima di costruire i `group` per LightGBM/CatBoost.
6. NaN significa "non applicabile", non 0: non imputare per i modelli ad alberi.
7. Un solo cambiamento per esperimento; ogni esperimento (anche fallito) va in `EXPERIMENTS.md` con CV media±std e comando.
8. Nessuna inferenza LLM sul test set.

## Fatti sui dati
- `task_name = <tipo>_<callsign>`; non esiste colonna `aircraft`: ricavarla dal suffisso (i `conflict_resolution_` hanno due callsign).
- `task_index` = tipo del task (14 classi). `seq_len` = numero di task nello snapshot.
- Non usare come feature: `sample_id, row_index, target_index, session, atco, scenario, timestamp, task_name` grezzi.
- Il codice nel file `Analisi Tecnica…md` è illustrativo: verificare sempre sui dati prima di copiarlo.

## Comandi
- Setup: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`
- Parquet (i CSV vanno messi in `data/`): `.venv/bin/python -m src.data --convert`
- Test: `.venv/bin/python -m pytest -q tests`
- Baseline + confronto con scorer: `.venv/bin/python -m src.baseline`
- CV (salva OOF e aggiunge la riga in EXPERIMENTS.md): `.venv/bin/python -m src.cv --features <vN> --exp-id <id> --note "<idea>"`
- Sanity metrica: `.venv/bin/python scorer.py --labelled data/validation.csv <submission.csv>`
- Submission (M8, da creare): `python -m src.make_submission --models <...> --out experiments/sub/<nome>.csv`
- Baseline |time_to_sector|: 0.307 su validation (0.35 è il valore sul test).

## Stile di lavoro
- Lavora una milestone alla volta; alla fine di ognuna riporta ≤10 righe con numeri e aspetta l'ok sul gate.
- Se una milestone supera 2× il tempo stimato, fermati e chiedi se ridurre l'ambito.
- Commit solo a gate superato; messaggi brevi in italiano.
