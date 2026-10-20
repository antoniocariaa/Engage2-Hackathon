# Guida all'implementazione agentica (1 persona + 1 agente Claude Code)

Obiettivo: eseguire `PIANO_TECNICO.md` milestone per milestone con un singolo agente, restando in controllo e senza perdere tempo in gara.

## 1. Modello di lavoro

- **Un agente, sequenziale**, una milestone (M0…M8) alla volta. Niente subagenti in parallelo: con 1 persona e CPU limitata il collo di bottiglia è il tempo di calcolo, non la scrittura del codice.
- Tu fai da **approvatore ai gate**: l'agente propone, esegue, riporta numeri; tu decidi se passare alla milestone successiva.
- Una **branch per milestone** (`m3-features`, …) e un commit a ogni gate superato, così si può tornare indietro se un'idea peggiora la CV.
- Il contesto dell'agente è `CLAUDE.md` (regole fisse) + `PIANO_TECNICO.md` (piano) + `experiments/EXPERIMENTS.md` (memoria dei risultati). Se la sessione si azzera, ripartire da lì.

## 2. Il ciclo di ogni milestone

1. **Leggi**: l'agente rilegge la sezione della milestone in `PIANO_TECNICO.md` e le ultime righe di `EXPERIMENTS.md`.
2. **Proponi**: breve piano (file toccati, esperimenti, tempo stimato). Tu approvi o correggi.
3. **Implementa + testa**: codice piccolo, funzioni pure, almeno un test per le parti a rischio (metrica, gruppi, leakage).
4. **Misura**: eseguire la CV GroupKFold per `atco`; riportare media, deviazione standard e score per ATCO.
5. **Gate**: confronto col criterio del piano. Superato → commit + riga in `EXPERIMENTS.md`. Non superato → annotare comunque (anche i fallimenti) e decidere: iterare, scartare o saltare.
6. **Rapporto**: l'agente riassume in ≤10 righe: cosa ha fatto, numeri, rischi, prossima mossa.

Regola di tempo: se una milestone supera di 2× la stima, fermarsi e chiedere se tagliare l'ambito.

## 3. Cosa far fare all'LLM e cosa no

| Sì | No |
|---|---|
| Scrivere codice, test, script di ablation | Inferenza LLM sulle righe del test set |
| Proporre feature di dominio dal dizionario FEATURES | Tuning guidato dalla public LB |
| Analisi degli errori su snapshot sbagliati | Decisioni di modello senza numeri di CV |
| Riassumere esperimenti in EXPERIMENTS.md | Modificare `scorer.py` o i CSV |

## 4. Prompt pronti, con livello di complessità e modello consigliato

Ogni prompt assume che `CLAUDE.md` sia già letto dall'agente. Sostituire i segnaposto `<…>`.
Struttura fissa dei prompt: **Contesto → Compito → Vincoli → Output atteso → Gate**. Il blocco "Output atteso" evita rapporti vaghi e rende facile il controllo.

### 4.1 Livelli di complessità

| Tag | Cosa significa | Modello consigliato | Quando |
|---|---|---|---|
| 🟢 **EASY** | Meccanico, un solo file/comando, esito verificabile a colpo d'occhio, nessuna decisione di design | Haiku 5.5 (`claude-haiku-5-5`) | Setup, log, formattazione, controlli di formato |
| 🟡 **MEDIUM** | Codice nuovo ma ben specificato, poche decisioni, test chiaro | Sonnet 5.5 (`claude-sonnet-5-5`) | Metrica, harness CV, wrapper di modelli, submission |
| 🟠 **HARD** | Richiede ragionamento di dominio o design con trade-off; errori sottili possibili ma rilevabili con la CV | Opus 5.5 (`claude-opus-5-5`) | Feature engineering, ensemble, pesi per episodio |
| 🔴 **VERY HARD** | Ragionamento multi-step su dati e codice insieme, ipotesi da falsificare, rischio di errori silenziosi (leakage, bug nei gruppi) | Fable 5.1 (`claude-fable-5-1`) | Analisi degli errori, audit di leakage, DeepSets |

Regole d'uso:
- **Escalation**: se un prompt fallisce il gate due volte con il modello consigliato, risali di un livello (Haiku → Sonnet → Opus → Fable). Non insistere con lo stesso modello.
- **Discesa**: se un prompt HARD si riduce a un passo meccanico (es. "aggiungi queste 5 colonne già decise"), eseguilo con Sonnet.
- Cambiare modello in Claude Code: `/model` prima di incollare il prompt.
- Indicativamente il grosso del tempo-modello va su 🟠 (M3). Risparmia sui 🟢/🟡 per avere margine di quota nelle ultime ore.

### 4.2 Indice rapido

| Prompt | Milestone | Livello | Modello |
|---|---|---|---|
| P0 Avvio sessione | — | 🟢 EASY | Haiku 5.5 |
| P1 Setup ambiente e parquet | M0 | 🟢 EASY | Haiku 5.5 |
| P2 Metrica e baseline | M1 | 🟡 MEDIUM | Sonnet 5.5 |
| P3 Harness CV | M2 | 🟡 MEDIUM | Sonnet 5.5 |
| P4 Famiglia di feature | M3 | 🟠 HARD | Opus 5.5 |
| P5 Brainstorming di feature di dominio | M3 | 🟠 HARD | Opus 5.5 |
| P6 Analisi degli errori | M3–M6 | 🔴 VERY HARD | Fable 5.1 |
| P7 Nuovo modello di ranking | M4 | 🟡 MEDIUM | Sonnet 5.5 |
| P8 Pesi per episodio | M5 | 🟠 HARD | Opus 5.5 |
| P9 Ensemble e post-processing | M6 | 🟠 HARD | Opus 5.5 |
| P10 DeepSets CPU | M7 | 🔴 VERY HARD | Fable 5.1 |
| P11 Submission e validazione formato | M8 | 🟡 MEDIUM | Sonnet 5.5 |
| P12 Audit leakage / CV anomala | qualsiasi | 🔴 VERY HARD | Fable 5.1 |
| P13 Aggiornamento log esperimenti | qualsiasi | 🟢 EASY | Haiku 5.5 |
| P14 Eseguire un esperimento su un altro PC | qualsiasi | 🟢 EASY | Haiku 5.5 / Sonnet 5.5 |

---

### P0 — Avvio sessione · 🟢 EASY · Haiku 5.5
```
CONTESTO: nuova sessione sul progetto AWARE.
COMPITO: leggi CLAUDE.md, PIANO_TECNICO.md e le ultime 10 righe di experiments/EXPERIMENTS.md (se esiste).
VINCOLI: non scrivere né eseguire codice.
OUTPUT ATTESO (max 5 righe): milestone corrente, ultimo CV score registrato (media±std), prossimo gate,
eventuali esperimenti lasciati a metà, branch git attuale.
```

### P1 — Setup ambiente e parquet (M0) · 🟢 EASY · Haiku 5.5
```
CONTESTO: M0 di PIANO_TECNICO.md. Solo CPU, ~7 GB di RAM libera, Python 3.12, nessuna libreria ML installata.
COMPITO: crea .venv; scrivi requirements.txt con versioni fissate (pandas, pyarrow, numpy, scikit-learn, lightgbm,
xgboost, catboost, scipy, pytest) e installa; converti train/validation/test.csv in data/*.parquet
(float32 dove possibile, task_name come stringa); aggiungi data/, .venv/, experiments/oof/, experiments/sub/ a .gitignore.
VINCOLI: non modificare i CSV originali; non committare dati.
OUTPUT ATTESO: versioni installate, tempo di lettura di ogni parquet, dimensioni su disco, shape di ogni file.
GATE: `import lightgbm, catboost, xgboost` ok; ogni parquet si carica in < 5 s.
```

### P2 — Metrica e baseline (M1) · 🟡 MEDIUM · Sonnet 5.5
```
CONTESTO: M1. La metrica ufficiale è in scorer.py (top-3 hit rate; duplicati rimossi; solo i primi 3 id distinti).
COMPITO: implementa src/metric.py con top3_hit_rate(df, score_col) e top3_by_group(df, score_col, group_col)
(per-ATCO). Genera la baseline "ordina per abs(time_to_sector) crescente" su validation.csv, scrivi il CSV di submission
in formato sample_id,prediction, e confronta con `python scorer.py --labelled validation.csv <file>`.
VINCOLI: nessuna modifica a scorer.py; gestire pareggi in modo deterministico (ordina anche per row_index).
OUTPUT ATTESO: i due numeri (metrica locale vs scorer.py) affiancati + test pytest che li confronta.
GATE: coincidono esattamente; baseline ≈ 0.35 (se molto diversa, segnala e indaga prima di proseguire).
```

### P3 — Harness CV (M2) · 🟡 MEDIUM · Sonnet 5.5
```
CONTESTO: M2. Dati = train+validation concatenati; gruppi di CV = atco; modello base = LightGBM lambdarank.
COMPITO: src/cv.py con GroupKFold(5) per atco; src/models/lgbm_rank.py (feature grezze + task_index categorico,
eval_at=[1,3], seed fisso); salva OOF in experiments/oof/<exp_id>.parquet (sample_id,row_index,score,fold,atco).
VINCOLI: righe ordinate per sample_id contiguo prima di calcolare group = conteggi per sample_id.
Escludi dalle feature: sample_id, seq_len, row_index, target_index, session, atco, scenario, timestamp, task_name.
Scrivi test pytest per: (a) allineamento gruppi/righe, (b) nessun atco in train e valid dello stesso fold,
(c) nessuna colonna vietata tra le feature.
OUTPUT ATTESO: tabella score per fold e per ATCO, media±std, tempo di esecuzione, riga in EXPERIMENTS.md.
GATE: CV ≥ baseline + 0.10; tutti i test passano.
```

### P4 — Famiglia di feature (M3) · 🟠 HARD · Opus 5.5
```
CONTESTO: M3. Feature set corrente = <vN-1> con CV <media±std>. Famiglia da aggiungere: <una tra:
(1) scomposizione time_to_sector; (2) contesto di snapshot (rank, pct_rank, z-score, diff dal min/max);
(3) carico dello snapshot; (4) aircraft ricostruito dal task_name e feature intra-aereo; (5) storia ATCO a_*>.
COMPITO: implementa SOLO questa famiglia in src/features.py come feature set <vN> = <vN-1> + nuove.
Per (4): estrai il callsign dal suffisso di task_name (conflict_resolution_* ha due callsign: crea una colonna per ciascuno
o una chiave canonica), poi groupby(sample_id, callsign): conteggio task, flag di co-occorrenza
(assume+entry, transfer+exit, zone_conflict), differenza/rapporto di x_task_age e x_state_age rispetto ai fratelli, rank intra-aereo.
VINCOLI: funzioni pure; mai usare target_index; NaN preservati (nessuna imputazione); float32; nessun cambio di
iperparametri nello stesso esperimento. Aggiungi un test: azzerando target_index le feature non cambiano.
OUTPUT ATTESO: CV <vN> vs <vN-1> (media±std), n. ATCO migliorati/peggiorati, top-15 importanze, 3 snapshot d'esempio
con le nuove feature, nota sul tempo di calcolo delle feature.
GATE: se non migliora la CV media E in almeno ~60% degli ATCO, non committare: annota il fallimento in EXPERIMENTS.md.
```

### P5 — Brainstorming di feature di dominio (M3) · 🟠 HARD · Opus 5.5
```
CONTESTO: abbiamo il dizionario delle 87 feature nella sezione 4 di "Analisi Tecnica Sfida SESAR AWARE.md" e il
feature set <vN> in src/features.py. Ruolo: controllore ATC en-route che sceglie quale task trattare adesso.
COMPITO: proponi 10 feature di dominio NON ancora presenti. Per ognuna: nome, formula, colonne usate,
ipotesi comportamentale (perché il controllore sceglierebbe questo task), rischio di leakage, costo di calcolo.
Ordina per rapporto (guadagno atteso / costo). Controlla sui dati (statistiche descrittive su train) che le colonne
usate abbiano abbastanza varianza e non siano costanti.
VINCOLI: nessuna implementazione, nessuna modifica ai file. Niente feature che usano target_index o informazioni del futuro
(timestamp successivi allo snapshot).
OUTPUT ATTESO: tabella di 10 righe + le 3 che implementeresti per prime e perché.
```

### P6 — Analisi degli errori (M3–M6) · 🔴 VERY HARD · Fable 5.1
```
CONTESTO: OOF dell'esperimento <exp_id> in experiments/oof/. Vogliamo capire dove il modello sbaglia, non solo quanto.
COMPITO: (1) calcola hit-rate per task_index del target, per fascia di seq_len e per ATCO; (2) estrai 30 snapshot in cui il
target NON è nel top-3, campionati su ≥10 ATCO diversi; per ciascuno mostra tipo del target, sua posizione, i 3 task predetti
e le feature più discriminanti; (3) separa gli errori in "ambiguità reale" (due task quasi equivalenti: stesso aereo, età simile)
vs "feature mancante" vs "bug"; (4) per ogni pattern ricorrente proponi una feature o un post-processing e stima
(sui soli OOF, senza riaddestrare) il guadagno massimo possibile.
VINCOLI: non modificare codice o dati; non basarti su singoli aneddoti: ogni pattern deve riguardare ≥50 snapshot o essere scartato.
OUTPUT ATTESO: elenco di max 5 pattern ordinati per snapshot recuperabili, con evidenza numerica e proposta concreta; indica
quanta parte dell'errore ritieni irriducibile (rumore inter-rater).
```

### P7 — Nuovo modello di ranking (M4) · 🟡 MEDIUM · Sonnet 5.5
```
CONTESTO: M4. Esiste src/models/lgbm_rank.py con interfaccia fit/predict e stessi fold di cv.py. Feature set <vN>.
COMPITO: aggiungi src/models/<cat_rank|xgb_rank>.py con la stessa interfaccia:
CatBoost loss QuerySoftMax (poi YetiRank), group_id = sample_id, task_index come categorico, thread_count=8;
oppure XGBoost rank:ndcg. Early stopping per fold, seed fisso.
VINCOLI: stessi fold e stesso feature set di LightGBM per confronto equo; tuning solo di 2–3 iperparametri; limita il
tempo (budget massimo per fold: <minuti>) e riduci iterations se supera.
OUTPUT ATTESO: CV media±std e per ATCO, correlazione tra OOF del nuovo modello e di LightGBM, tempo per fold.
GATE: CV comparabile al miglior GBDT (±0.01) e OOF salvati; correlazione < 0.95 è un buon segno per l'ensemble.
```

### P8 — Pesi per episodio (M5) · 🟠 HARD · Opus 5.5
```
CONTESTO: M5. ~15k sample derivano da ~4150 episodi di lavoro (task di 5–40 s) campionati ogni 3 s: i task lunghi dominano la loss.
COMPITO: (1) ricostruisci gli episodi: snapshot consecutivi della stessa session in cui il target ha lo stesso task_name;
verifica che il numero di episodi sia vicino a ~4150 e che le lunghezze stiano in 5–40 s (≈2–13 snapshot);
(2) calcola peso riga = 1/len^alpha con alpha ∈ {0, 0.5, 1} e applicalo ai soli sample del gruppo (non alle righe negative separatamente);
(3) confronta la CV.
VINCOLI: gli episodi sono usati SOLO per i pesi di training, mai come feature; la valutazione resta non pesata (la metrica ufficiale
pesa ogni sample ugualmente); le righe di uno stesso episodio devono restare nello stesso fold (lo sono già, perché il fold è per atco).
OUTPUT ATTESO: statistiche degli episodi ricostruiti, CV per alpha (media±std, mediana per ATCO), raccomandazione.
GATE: adottare solo se migliora media E mediana per ATCO; altrimenti scartare e annotare.
```

### P9 — Ensemble e post-processing (M6) · 🟠 HARD · Opus 5.5
```
CONTESTO: M6. Abbiamo OOF di ≥2 modelli (stessi fold) in experiments/oof/.
COMPITO: src/ensemble.py: (1) normalizza gli score di ogni modello con softmax intra-snapshot (temperatura stimata sugli OOF)
e in alternativa con rank-normalization; (2) cerca i pesi con griglia semplice o scipy.optimize su CV annidata
(pesi stimati su 4 fold e valutati sul 5°, a rotazione) per evitare ottimismo; (3) testa il post-processing "diversificazione top-3"
(se i primi due sono sullo stesso aereo, prova a spostare un task di un altro aereo in 3ª posizione) SOLO sugli OOF.
VINCOLI: nessun uso della public LB; ogni scelta deve avere guadagno ≥ 1 deviazione standard tra ATCO o essere scartata.
OUTPUT ATTESO: CV ensemble vs miglior singolo, pesi scelti, effetto del post-processing, versione "semplice" (media uguale) come riserva.
GATE: ensemble > miglior singolo in CV nested; salva i pesi in experiments/ensemble_<id>.json.
```

### P10 — DeepSets CPU (M7) · 🔴 VERY HARD · Fable 5.1
```
CONTESTO: M7, SOLO se M6 è chiuso e restano ≥4 ore. PyTorch CPU (installa torch cpu), 8 core, feature set <vN> (dalle feature
di snapshot già calcolate, quindi il modello non deve riscoprirle).
COMPITO: implementa src/models/deepsets.py: input = tensore [n_snapshot, 59, n_feat] con maschera di padding; NaN → 0 + canale
maschera per colonna; normalizzazione per feature stimata solo sul training del fold; blocchi di self-attention senza positional
encoding (d_model=64, 2 blocchi, 4 teste) o DeepSets (phi → media/max pooling → rho concatenato per riga); loss = cross-entropy
softmax sull'intero snapshot (logit mascherati); AdamW, early stopping per fold, seed fissi, torch.set_num_threads(8).
VINCOLI: tempo massimo per fold <minuti>; se un fold supera il budget riduci epoche/dimensioni; deve produrre OOF compatibili con
ensemble.py; verifica con un test di permutation-equivariance (permutando le righe di uno snapshot, i logit si permutano uguali).
OUTPUT ATTESO: CV media±std, tempo per fold, contributo marginale all'ensemble (CV con e senza), curva di training.
GATE: contributo positivo all'ensemble in CV nested; altrimenti scarta e annota. Non ritardare M8 per questo.
```

### P11 — Submission e validazione formato (M8) · 🟡 MEDIUM · Sonnet 5.5
```
CONTESTO: M8. Modelli e pesi scelti in <exp_id/ensemble_id>. Le iterazioni finali = media delle best_iteration dei fold.
COMPITO: riaddestra su train+validation, predici test.csv, scrivi experiments/sub/<nome>.csv con
src/make_submission.py (colonne sample_id,prediction; 3 row_index distinti separati da spazio, migliore per primo).
Il check deve fallire se: sample_id ≠ 1863 unici o mancanti, una riga ha <3 id distinti, un id non appartiene a quello snapshot,
ci sono NaN negli score.
Sanity aggiuntiva: stessa pipeline addestrata SOLO su train → scorer.py --labelled validation.csv.
VINCOLI: nessuna modifica ai modelli; salva anche gli score grezzi per eventuali blend successivi.
OUTPUT ATTESO: percorso del file, risultato del check, score di sanity, submission rimaste oggi (max 5/giorno, reset 02:00 Zagabria).
```

### P12 — Audit di leakage / CV anomala · 🔴 VERY HARD · Fable 5.1
```
CONTESTO: usare quando la CV è troppo bella (> ~0.8) o molto diversa dalla public LB, o un'ultima modifica ha un salto sospetto.
COMPITO: audit completo di <exp_id>: (1) verifica che nessuna feature derivi da target_index, direttamente o via groupby/transform/rank
che includa la riga target; (2) verifica che i fold siano per atco e che nessuna sessione compaia in più fold; (3) verifica che feature
globali (normalizzazioni, encoding, target encoding) siano calcolate dentro il fold; (4) prova sperimentale: rimescola target_index
dentro ogni snapshot e verifica che la CV crolli al livello del caso (≈ 3/seq_len medio); (5) elimina le feature a più alta importanza
una alla volta e osserva il calo; (6) confronta la distribuzione degli score OOF per ATCO.
VINCOLI: non correggere nulla finché non hai individuato la causa con una prova; riporta le prove, non solo le conclusioni.
OUTPUT ATTESO: verdetto (leakage sì/no), causa precisa con riga di codice, correzione proposta, CV dopo la correzione.
```

### P13 — Aggiornamento log esperimenti · 🟢 EASY · Haiku 5.5
```
CONTESTO: esperimento <exp_id> appena concluso.
COMPITO: aggiungi una riga a experiments/EXPERIMENTS.md nel formato: id | data/ora | milestone | idea | feature set | modello |
CV media±std | min–max per ATCO | note/decisione. Incolla il comando esatto usato.
VINCOLI: non modificare le righe precedenti; non inventare numeri: se un dato manca scrivi "n/d".
OUTPUT ATTESO: la riga aggiunta.
```

### P14 — Eseguire un esperimento su un altro PC · 🟢 EASY · Haiku 5.5 / Sonnet 5.5
```
CONTESTO: repo AWARE (branch <branch>), su un PC diverso da quello di sviluppo. I CSV della competizione
(train.csv, validation.csv, test.csv, sample_submission.csv) sono in <percorso>.
COMPITO:
1. git clone <url-repo> && cd Engage2-Hackathon && git checkout <branch>
2. copia i 4 CSV in data/
3. rileva l'hardware: numero di core (nproc / os.cpu_count) e se c'è una GPU NVIDIA funzionante (nvidia-smi)
4. esegui: AWARE_THREADS=<core> ./run.sh <modello> <feature_set> <exp_id>
   (es. ./run.sh cat_rank ac m4-cat-ac-pc2). Se c'è una GPU NVIDIA e il modello è cat_rank, prova prima con AWARE_GPU=1;
   se CatBoost dà errore sulla GPU (loss non supportata, driver), rilancia senza AWARE_GPU.
   Su Windows senza bash: usa WSL o Git Bash; in alternativa esegui a mano i comandi di run.sh con .venv\Scripts\python.
VINCOLI: non modificare il codice né i parametri; non fare commit o push; se i test falliscono fermati e riporta l'errore.
Usa un exp_id con il nome del PC (es. m4-cat-ac-pc2) per non sovrascrivere i file del PC di sviluppo.
PREREQUISITO (sul PC di sviluppo): il branch deve essere committato e pushato, altrimenti l'altro PC scarica codice vecchio.
OUTPUT ATTESO: hardware usato, tempo totale, riga "CV hit3 = ..." e score per ATCO; poi copia sul PC di sviluppo
experiments/oof/<exp_id>.parquet, experiments/oof/<exp_id>.json e l'ultima riga di experiments/EXPERIMENTS.md.
```

## 5. Controlli che l'agente deve fare sempre (anti-errore)

1. **Numeri prima delle opinioni**: ogni affermazione "migliora" deve avere CV media±std e ATCO migliorati/peggiorati.
2. **Leakage check**: nessuna feature derivata da `target_index`; split solo per `atco`.
3. **Riproducibilità**: seed fissi, comando esatto salvato in `EXPERIMENTS.md`.
4. **Un cambiamento alla volta** nelle ablation (feature set *oppure* parametri, mai entrambi).
5. **Non toccare** `scorer.py`, `train/validation/test.csv`, `sample_submission.csv`.
6. **Non fidarsi del documento alla lettera**: i frammenti di codice in `Analisi Tecnica…md` sono illustrativi (es. usano una colonna `aircraft` che non esiste); verificare sempre sui dati.

## 6. Formato di `experiments/EXPERIMENTS.md`

```
| id | data/ora | milestone | idea | feature set | modello | CV mean ± std | min–max per ATCO | note / decisione |
```
Una riga per esperimento, inclusi i fallimenti.

## 7. Cadenza consigliata (≈20 h)

| Blocco | Milestone | Ore |
|---|---|---|
| Sera 20 ott | M0–M2 | 2 |
| Sera 20 ott | M3 (iterazioni su feature) | 3 |
| Notte (calcoli lunghi in background) | M4 tuning/CatBoost, mentre dormi/riposi | 3 |
| Mattina 21 ott fino alle 10:30 | M5–M6 | 3 |
| 10:30–12:15 | M8 submission finali + selezione | 1.5 |
| 12:15–12:45 | buffer, **nessuna modifica nuova** | — |
| Solo se avanza tempo | M7 DeepSets | — |

Uso submission suggerito: #1 (dopo M2, sanity), #2 (dopo M3), #3 (ensemble), #4–5 riserva. Le 2 finali si scelgono sulla CV, non sulla public.

## 8. Dopo la gara (intervista top-3)

Se arrivi nei primi 3 c'è un colloquio tecnico. `EXPERIMENTS.md` + `PIANO_TECNICO.md` sono la tua traccia: tienili aggiornati per spiegare scelte di validazione (GroupKFold per ATCO), feature chiave e perché la public LB non è stata usata per il tuning.
