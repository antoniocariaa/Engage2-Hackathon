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

## 4. Prompt pronti (copia/incolla)

Sostituire `<…>`. Ogni prompt assume che `CLAUDE.md` sia già letto dall'agente.

### P0 — Avvio sessione
```
Leggi CLAUDE.md, PIANO_TECNICO.md e experiments/EXPERIMENTS.md.
Dimmi in 5 righe: a che milestone siamo, qual è l'ultimo CV score registrato, e qual è il prossimo gate.
Non scrivere codice finché non confermo.
```

### P1 — Milestone M0/M1 (setup, metrica, baseline)
```
Esegui M0 e M1 di PIANO_TECNICO.md.
Crea .venv, installa le dipendenze CPU, converti i CSV in data/*.parquet (gitignored).
Implementa src/metric.py e un test che confronta top3_hit_rate con scorer.py su validation.csv
(usa una submission generata dalla baseline |time_to_sector| crescente).
Mostrami i due numeri affiancati. Gate: coincidono esattamente, baseline ≈ 0.35.
```

### P2 — M2 (harness CV)
```
Esegui M2. Implementa src/cv.py: GroupKFold(5) per atco su train+validation (concatenati),
modello LightGBM lambdarank sulle sole feature grezze + task_index categorico.
Requisiti: righe ordinate per sample_id, group = conteggi per sample_id, test unit che verifica
l'allineamento dei gruppi e che nessun atco compare in train e valid dello stesso fold.
Salva OOF in experiments/oof/. Stampa score per fold e per ATCO. Aggiorna EXPERIMENTS.md.
```

### P3 — M3 (feature engineering) — ripetibile per ogni famiglia di feature
```
Esegui il punto <N> di M3 (<nome famiglia: es. "aircraft ricostruito dal task_name">).
Aggiungi le feature a src/features.py come nuova versione del feature set (<vX> = <vX-1> + nuove).
Vincoli: funzioni pure, nessun uso di target_index, NaN preservati, float32.
Poi lancia la CV con <vX> e confronta con <vX-1>: media±std e numero di ATCO migliorati/peggiorati.
Mostra le 15 feature più importanti. Se non migliora, dimmelo e non committare.
```

### P4 — Brainstorming di feature (uso LLM come co-pilota di dominio)
```
Leggi la sezione 4 di "Analisi Tecnica Sfida SESAR AWARE.md" (le 87 feature) e le colonne di data/train.parquet.
Proponi 10 feature di dominio ATC non ancora in features.py, ciascuna con: formula, colonne usate,
perché dovrebbe discriminare il task scelto dal controllore, rischio di leakage.
Ordinale per rapporto (atteso guadagno / costo). Non implementare nulla ancora.
```

### P5 — Analisi degli errori
```
Usando gli OOF di <exp_id>, estrai 20 snapshot in cui il target NON è nel top-3 (distribuiti su ATCO diversi).
Per ciascuno mostra: tipo del target, sua posizione nel ranking, i 3 task predetti, e le feature principali che differiscono.
Raggruppa in pattern ricorrenti (es. "target = transfer_aircraft preceduto da exit_coordination dello stesso aereo")
e proponi per ogni pattern una feature o un post-processing. Non modificare codice.
```

### P6 — M4 (modelli), M6 (ensemble)
```
Esegui M4: aggiungi src/models/cat_rank.py (CatBoost QuerySoftMax, task_index categorico, group_id=sample_id)
con la stessa interfaccia di lgbm_rank.py. Stessi fold, salva OOF. Confronta con LightGBM.
Poi M6: src/ensemble.py normalizza gli score con softmax intra-snapshot, cerca i pesi sugli OOF
(griglia semplice, non ottimizzatori complessi) e riporta CV dell'ensemble vs miglior singolo.
```

### P7 — M8 (submission)
```
Esegui M8. Riaddestra <modelli> su train+validation con le iterazioni medie trovate in CV,
predici test.csv, scrivi experiments/sub/<nome>.csv con make_submission.py.
Il check deve verificare: 1863 sample_id unici e completi, 3 row_index distinti per riga, tutti presenti nello snapshot.
Stampa anche lo score di scorer.py --labelled validation.csv su una versione ottenuta con il modello addestrato SOLO su train
(solo come sanity). Dimmi cosa caricare e quante submission restano oggi.
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
