# Piano Tecnico di Implementazione — AWARE Task-Prioritisation (Engage #2)

Derivato da `Analisi Tecnica Sfida SESAR AWARE.md`, adattato ai vincoli reali del team:
**1 persona, PC consumer solo CPU (8 core, ~15 GB RAM), ~20 ore, 5 submission/giorno, chiusura mer 21 ott 13:00 (Zagabria).**

Questo documento dice *cosa* costruire. Il *come* operativo con gli agenti è in `GUIDA_AGENTICA.md`; le regole fisse per l'agente sono in `CLAUDE.md`.

---

## 1. Problema in una riga

Per ogni `sample_id` (snapshot ogni 3 s) c'è una lista di 9–59 task aperti (righe). Esattamente una riga ha `target_index=1`. Bisogna ordinare le righe e restituire i primi 3 `row_index`. Metrica: **top-3 hit rate** (`scorer.py`). Baseline organizzatori = 0.35.

È un problema di **ranking per gruppo**, non di classificazione.

## 2. Fatti verificati sui dati (non dal documento)

- `task_name` = `<tipo>_<callsign>` (es. `assume_aircraft_SAS77A`, `conflict_resolution_QTR50U_GFM7Z`). **Non esiste una colonna `aircraft`**: va ricostruita dal suffisso di `task_name`. Il codice d'esempio del documento (`if "aircraft" in out.columns`) altrimenti non farebbe nulla.
- `task_index` = id del tipo di task (14 classi, es. 2=entry_coordination, 4=assume, 3=exit_coord, 5=transfer, 12=zone_conflict).
- Task fratelli (stesso aereo) hanno quasi tutte le feature identiche → servono feature che le distinguano (`x_task_age`, `x_state_age`, gruppo `l_*`, tipo).
- `sample_id` di validation parte da 11923: gli id sono globali tra i file.
- Colonne identificative da **non** usare come feature: `sample_id`, `seq_len`, `row_index`, `target_index`, `session`, `atco`, `scenario`, `timestamp`, `task_name` (grezzo). `atco` serve solo per i gruppi di CV.
- Ambiente: nessuna libreria ML installata (solo numpy). Python 3.12.

## 3. Split e validazione (la parte che decide la vittoria)

| Set | Contenuto | Uso |
|---|---|---|
| train.csv | 34 sessioni, ATCO 001–018 + 019sc3 | sviluppo |
| validation.csv | 019sc1, 019sc2, 020sc1 | CV + training finale |
| test.csv | public = 020sc2/sc3 (915), private = 023 (948) | solo predizione |

Regole:
1. **CV locale = `GroupKFold` per `atco`** (5 fold) su train+validation. Mai split per riga o per sample (near-duplicate a 3 s → leakage di 25–45 punti).
2. La **Public LB è solo un controllo di integrità** (ATCO 020 ≠ ATCO 023, il private è su un controllore mai visto). Non ottimizzare iperparametri sulla public.
3. Metrica locale: la stessa di `scorer.py`. Funzione unica in `src/metric.py`, testata contro `scorer.py` su validation.
4. Salvare sempre le **predizioni OOF** (per blending) con gli stessi fold.
5. Si può anche riportare il punteggio per-ATCO: la varianza tra controllori è il vero rumore (tetto teorico basso, accordo inter-rater ~50% su coppie vicine). Un miglioramento < deviazione standard tra ATCO non è un miglioramento.

## 4. Architettura del codice

```
src/
  config.py        # path, seed, colonne, fold
  data.py          # load csv (float32, category), split helpers
  metric.py        # top3_hit_rate(df, score_col), per-atco
  features.py      # build_features(df) -> df  (solo funzioni pure, no stato)
  cv.py            # GroupKFold per atco, salva oof
  models/
    lgbm_rank.py   # lambdarank / rank_xendcg
    cat_rank.py    # CatBoost QuerySoftMax / YetiRank
    deepsets.py    # opzionale, PyTorch CPU
  ensemble.py      # blend pesi su OOF
  make_submission.py  # scrive submission.csv + check formato
experiments/
  EXPERIMENTS.md   # log: id, idea, feature set, CV mean±std, note
  oof/ , sub/      # file generati (gitignored)
```

Principio: tutto riproducibile con `python -m src.cv --model lgbm_rank --features v2` e seed fissi.

## 5. Milestone (con gate di uscita)

Tempi indicativi per ~20 h, CPU. Ogni gate è un criterio *misurabile*: non si passa avanti se non è soddisfatto.

### M0 — Ambiente (≈20 min)
- `python -m venv .venv`; installare `pandas pyarrow numpy scikit-learn lightgbm xgboost catboost scipy` (CPU). PyTorch CPU solo se si arriva a M7.
- Convertire i CSV in **parquet** (train 119 MB → molto più veloce da ricaricare) in `data/`, gitignored.
- **Gate**: `import lightgbm, catboost, xgboost` ok; parquet letti in < 5 s.

### M1 — Baseline e metrica (≈40 min)
- Implementare `src/metric.py`. Riprodurre la baseline `ordina per |time_to_sector| crescente` su validation.
- **Gate**: la metrica locale coincide **esattamente** con `python scorer.py --labelled validation.csv <sub>`; baseline ≈ 0.35.

### M2 — Harness CV (≈1 h)
- `GroupKFold(5)` per `atco` su train+validation; ritorna score per fold e per ATCO; salva OOF.
- Primo modello banale: LightGBM `lambdarank` sulle feature grezze (+ `task_index` categorico), `group` = righe per `sample_id` (le righe devono essere **ordinate per sample_id contiguo**).
- **Gate**: CV > baseline di almeno +0.10; pipeline end-to-end funziona; ordinamento gruppi verificato da un test.

### M3 — Feature engineering v1 (≈3 h) — il grosso del guadagno
Tutte calcolate con `groupby` e **senza mai usare `target_index`** (check automatico anti-leakage: nessuna feature dipende dal target).
1. **Scomposizione `time_to_sector`**: `abs_tts`, `is_inside`, `sec_to_entry` (tts≥0), `sec_to_exit` (tts<0).
2. **Contesto di snapshot** (`groupby sample_id`) su `abs_tts`, `x_task_age`, `x_state_age`, `k_timeToExitPoint`, `time_to_MD`, `distance_to_sector`: rank, pct_rank, z-score, differenza dal minimo/massimo.
3. **Carico dello snapshot**: `n_open` (= `seq_len`), conteggio per `task_index`, n. conflitti attivi, n. task in finestra critica (|tts| < soglia).
4. **Aircraft ricostruito**: `callsign = task_name` senza il prefisso tipo (gestire `conflict_resolution_A_B` con due callsign). Poi `groupby(sample_id, callsign)`: `ac_n_tasks`, flag co-occorrenza (`has_assume&entry`, `has_transfer&exit`, `has_zone_conflict`), differenza/rapporto di `x_task_age` e `x_state_age` rispetto agli altri task dello stesso aereo, rank intra-aereo.
5. **Storia ATCO**: `a_since_any`, `a_n60`, `a_n300` interagiti con tipo task; `a_is_last_cs` (ultimo aereo toccato) è probabilmente fortissimo → verificare importanza.
6. **NaN**: "vuoto ≠ 0" → lasciare `NaN` ai GBDT, non imputare.
7. Una funzione `build_features(df, version)` con feature set versionati (`v1`, `v2`…) per fare ablation.
- **Gate**: CV GroupKFold migliora di ≥ +0.05 su M2 e il miglioramento è presente nella maggioranza dei fold/ATCO.

### M4 — Modelli di ranking GBDT (≈3 h)
Ordine di priorità (tutti CPU-friendly):
1. LightGBM `lambdarank` (`eval_at=[1,3]`, `lambdarank_truncation_level` ~ 5–10) e `rank_xendcg` (più vicino a softmax listwise).
2. CatBoost `QuerySoftMax` (poi `YetiRank`), `group_id=sample_id`, `task_index` come categorico.
3. XGBoost `rank:ndcg` come terza vista se resta tempo.
- Tuning **leggero** (num_leaves, min_data_in_leaf, learning_rate, feature_fraction) con early stopping per fold. Tenere i modelli piccoli: 541k righe, 8 core.
- **Gate**: ≥ 2 modelli con CV comparabile e OOF salvati.

### M5 — Pesi per episodio (≈1 h, solo se misurabile)
- Ricostruire gli episodi: snapshot consecutivi della stessa `session` con lo stesso task target (`task_name` del target). Peso riga ∝ `1/sqrt(len_episodio)`.
- **Gate**: tenere solo se la CV (media e mediana per ATCO) migliora; altrimenti scartare e annotare in EXPERIMENTS.md.

### M6 — Ensemble e post-processing (≈1.5 h)
- Normalizzare gli score di ciascun modello con **softmax intra-snapshot** (o rank-normalization), poi media pesata; pesi stimati sugli OOF (non sulla public).
- Post-processing da testare solo su OOF: *diversificazione top-3* (se i primi due sono sullo stesso aereo, valutare se promuovere un terzo di un altro aereo) — da adottare solo se aumenta lo hit-rate.
- **Gate**: ensemble > miglior singolo modello in CV.

### M7 — (Opzionale) DeepSets / Set-Transformer CPU (≈3 h)
- Solo se M6 è chiuso e restano ≥ 4 h. PyTorch CPU, modello molto piccolo (d=64, 2 blocchi attention, senza positional encoding), padding/mask a 59 righe, loss = cross-entropy softmax sull'intero snapshot. Input = feature M3 normalizzate (NaN → 0 + maschera).
- **Gate**: contributo positivo all'ensemble in CV; altrimenti scartare. Priorità bassa: su CPU è il componente a più alto rischio/tempo.

### M8 — Submission finale (≈1 h)
- Riaddestrare i modelli scelti su **train+validation** con iterazioni fissate dalla CV; predire `test.csv`.
- `make_submission.py` produce `sample_id,prediction` (3 `row_index` separati da spazio) e **verifica**: 1863 sample_id unici, nessuno mancante, 3 id distinti e presenti nello snapshot.
- Budget: 5 submission/giorno, reset 02:00 Zagabria. Pianificare: (1) sanity pipeline, (2) singolo modello, (3) ensemble, tenerne 1–2 di riserva.
- Selezione finale (2 submission): **la migliore per CV GroupKFold**, non per public LB; la seconda = variante più robusta (es. ensemble più semplice).
- **Gate**: file valido, ordine di punteggio public coerente con la CV (differenze grandi = bug o leakage).

## 6. Ruolo dell'LLM / dell'agente

Come dice il documento: **nessuna inferenza LLM sulle righe** (541k righe, latenza e soglie numeriche). L'LLM serve in fase di sviluppo per:
- leggere il dizionario FEATURES e proporre feature di dominio (indici di urgenza di uscita, stato coordinamento);
- scrivere/rifattorizzare codice, test e script di ablation;
- fare analisi degli errori (guardare snapshot sbagliati e proporre feature mancanti).

Vedi `GUIDA_AGENTICA.md` per i prompt.

## 7. Rischi principali

| Rischio | Mitigazione |
|---|---|
| Leakage (split per riga/sample) | solo `GroupKFold(atco)`; test automatico che nessun `atco` stia in train e valid |
| Overfit su ATCO 020 / public LB | decisioni solo su CV; public = sanity check |
| Gruppi LightGBM disallineati | ordinare per `sample_id`, `group = counts` — test unit |
| Feature che usano il target | funzioni di `features.py` non ricevono `target_index`; test che le feature non cambiano se si azzera il target |
| CPU/RAM (15 GB, ~7 GB liberi) | `float32`, parquet, `category`, feature set versionati, niente copie di DataFrame inutili |
| Tempo | gate rigidi; M7 è sacrificabile, M3–M6 no |
| Submission invalida | validazione formato prima di ogni upload + `scorer.py --labelled` su validation |
