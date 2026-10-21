# AWARE: prevedere su quale compito sta lavorando il controllore

*Come siamo passati dal 35% al 93% di risposte corrette*

---

## In breve

- **Il problema.** Ogni 3 secondi il sistema AWARE mostra al controllore di volo una lista di circa 36 compiti aperti. Dobbiamo indovinare su quale sta lavorando, e possiamo dare **3 tentativi**.
- **Il risultato.** Siamo partiti dal **35%** della regola di base fornita dagli organizzatori. Oggi siamo al **93% in validazione** e al **94% sulla leaderboard pubblica**.
- **La svolta.** Il controllore lavora su un compito per alcuni secondi consecutivi (un *blocco*). Quando ha finito, quel compito **sparisce dalla lista proprio alla fine del blocco**. Leggendo gli istanti successivi della stessa sessione, che il file di test contiene, il modello lo "vede".
- **Il metodo.** Ogni miglioramento è stato misurato su controllori **mai visti** in addestramento, e adottato solo se migliorava in modo stabile.

---

## 1. Il problema in parole semplici

Un controllore del settore svedese `ESMM_23` gestisce gli aerei che attraversano il suo spazio aereo. Il sistema AWARE tiene una lista automatica dei compiti aperti. Per esempio:

| Compito | Significato |
|---|---|
| `assume_aircraft_SAS95T` | prendere in carico l'aereo SAS95T dal settore precedente |
| `transfer_aircraft_RYR1977` | passare l'aereo RYR1977 al settore successivo |
| `zone_conflict_FIN954` | la rotta di FIN954 attraversa una zona militare attiva |
| `exit_coordination_GFM7Z` | concordare con il settore successivo come uscirà GFM7Z |

Alcuni esperti hanno guardato le registrazioni delle simulazioni e hanno annotato, istante per istante, **quale compito stava svolgendo il controllore**.

- **Un campione** è una "fotografia" della lista in un certo istante: in media 36 compiti, da un minimo di 9 a un massimo di 59.
- **Il nostro compito** è indicare i 3 compiti più probabili. Se quello giusto è tra i 3, il campione vale 1 punto.
- **Il punteggio** (top-3 hit rate) è la percentuale di campioni indovinati. Tirando a caso si ottiene circa il 9%.

**I dati in numeri:**
- Addestramento: 34 sessioni, 16 controllori, 11.923 campioni, 432.578 righe e 86 colonne di informazioni per compito.
- Validazione: 2 controllori nuovi (019 e 020).
- Test: i controllori 020 (leaderboard pubblica) e 023 (leaderboard privata, mai visto).

---

## 2. Il percorso: da 0,09 a 0,93

| Passo | Cosa abbiamo fatto | CV (controllori mai visti) | Validazione |
|---|---|---:|---:|
| 0 | Scelta casuale di 3 compiti | 0,091 | 0,097 |
| 1 | Regola degli organizzatori (riprodotta esattamente) | 0,356 | 0,485 |
| 2 | Modello di machine learning (LightGBM) | 0,650 | 0,652 |
| 3 | Modello che impara a **ordinare** la lista (lambdarank), 3 semi | 0,658 | 0,647 |
| 4 | + **peso ridotto** agli istanti quasi identici | 0,661 | 0,661 |
| 5 | + **informazioni sui blocchi e sugli istanti successivi** | **0,901** | **0,929** |
| 6 | + cambi di stato attorno al blocco, media di due modelli (**finale**) | **0,906** | **0,930** |

La riga 5 ha ottenuto **0,94 sulla leaderboard pubblica**. La riga 6 è la submission finale; il suo punteggio in leaderboard non è ancora noto.

> "CV" è la media su 4 prove, ognuna con 4 controllori tenuti da parte. È la nostra stima più affidabile per un controllore nuovo. La sezione 4 spiega perché.

---

## 3. La pipeline, passo per passo

```
  CSV grezzi
     │
     ▼
 ① Controllo dei dati ─────────── i dati sono come dichiarato?
     │
     ▼
 ② Metrica verificata ─────────── il nostro punteggio coincide con quello ufficiale?
     │
     ▼
 ③ Validazione per controllore ── simuliamo "controllore mai visto"
     │
     ▼
 ④ Baseline ───────────────────── riprodurre la regola ufficiale: il punto di partenza
     │
     ▼
 ⑤ Feature ────────────────────── trasformare i dati in indizi utili
     │
     ▼
 ⑥ Modello di ranking ─────────── dare un punteggio a ogni compito e ordinarli
     │
     ▼
 ⑦ Selezione ──────────────────── tenere solo i miglioramenti stabili
     │
     ▼
 ⑧ Addestramento finale ───────── riaddestrare su tutti i dati etichettati
     │
     ▼
 ⑨ Submission validata ────────── 1.863 righe, 3 compiti validi per riga
```

### ① Controllo dei dati

Prima di costruire qualsiasi modello abbiamo verificato automaticamente che i dati fossero quelli dichiarati:
- il numero di sessioni, campioni e righe;
- **esattamente un compito corretto** per ogni campione;
- nessuna sessione presente in due file.

Tutto tornava. Abbiamo trovato solo piccole anomalie, per esempio qualche centinaio di "età" negative. Il report completo è in `reports/data_audit.md`.

### ② Metrica verificata

Abbiamo riscritto il calcolo del punteggio e controllato che desse **esattamente** lo stesso risultato dello script ufficiale `scorer.py`. Altrimenti rischieremmo di ottimizzare la cosa sbagliata.

### ③ Validazione per controllore: il concetto più importante

Il test contiene **controllori nuovi**, quindi anche la nostra valutazione deve usare controllori che il modello non ha mai visto.

```
16 controllori di addestramento, divisi in 4 gruppi da 4:

  prova 1:  addestra su [B C D]  →  valuta su [A]
  prova 2:  addestra su [A C D]  →  valuta su [B]
  prova 3:  addestra su [A B D]  →  valuta su [C]
  prova 4:  addestra su [A B C]  →  valuta su [D]
```

**Perché non dividere a caso?** Due fotografie a 3 secondi di distanza sono quasi identiche. Se una finisce in addestramento e l'altra in valutazione, il modello "copia" la risposta e il punteggio esce gonfiato.

### ④ Baseline: riprodurre la regola ufficiale

La regola degli organizzatori ordina i compiti per tipo e, a parità di tipo, mette prima l'aereo più vicino al confine del settore. L'abbiamo **ricostruita in modo esatto: 1.863 campioni di test su 1.863 identici** al file ufficiale.

Così abbiamo scoperto un dettaglio non scritto nella documentazione: i tipi sono ordinati per **tasso** di risposta corretta, non per frequenza.

### ⑤ Feature: gli indizi

Ogni compito arriva con 86 informazioni. Alcuni esempi:
- distanza e tempo dal confine del settore;
- conflitti previsti con altri aerei;
- stato del coordinamento con i settori vicini;
- da quanto tempo il controllore non dà istruzioni a quell'aereo;
- da quanto tempo il compito è in lista.

Il modello usa queste 86 informazioni più il **tipo di compito**. Le colonne "di servizio" sono **escluse di proposito**: identificativo del controllore, della sessione, l'orario. Il modello non deve poter imparare le abitudini di un singolo controllore.

### ⑥ Il modello: alberi decisionali che imparano a ordinare

- **Gradient boosting (LightGBM):** centinaia di piccoli alberi decisionali del tipo "se il compito è un transfer e l'aereo è a meno di 2 minuti dall'uscita, allora...". Ogni albero corregge gli errori dei precedenti. È lo standard per dati tabellari come questi.
- **Classificazione vs ranking:**
  - un modello *binario* impara "questo compito è quello giusto, sì o no?";
  - un modello di *ranking* (lambdarank) impara direttamente "**quale compito viene prima** nella stessa lista".

  Il nostro obiettivo è proprio ordinare, e il ranking ha funzionato meglio in tutte e 4 le prove.
- **Uscita:** un punteggio per ogni compito della lista. Ordiniamo e prendiamo i primi 3.

*Esempio illustrativo: i punteggi sono inventati a scopo didattico.*

```
Lista dell'istante t           punteggio      top-3
  transfer_aircraft_RYR1977       2,31    →  ①
  zone_conflict_FIN954            1,87    →  ②
  assume_aircraft_SAS95T          1,02    →  ③
  entry_coordination_SAS95T      -0,40
  ... (altri 32 compiti)
```

### ⑦ Selezione: adottare solo ciò che è davvero meglio

Abbiamo scoperto che **lo stesso modello, cambiando solo il seme casuale, oscilla tra 0,645 e 0,650**. Una differenza di 0,003 può essere solo fortuna. Per questo:
- ogni confronto usa **3 semi** mediati;
- una modifica è adottata solo se:
  1. migliora la media della CV, **e**
  2. migliora in almeno **3 prove su 4**, **e**
  3. non peggiora la validazione di più di 0,01.

Ogni esperimento, anche quelli falliti, è registrato in `reports/experiments.csv`.

### ⑧ Addestramento finale e ⑨ submission

- Il modello scelto viene riaddestrato su addestramento + validazione, con il numero di alberi fissato dalla CV.
- Le previsioni sul test vengono **rilette dal disco e controllate**: tutte le 1.863 righe presenti, 3 compiti distinti e validi per riga.

---

## 4. Due idee che hanno fatto la differenza

### Idea 1: non contare due volte la stessa cosa (peso per "corsa")

Quando il controllore lavora 30 secondi sullo stesso compito, abbiamo 10 fotografie quasi uguali con la stessa risposta. Senza correzione, gli episodi lunghi dominano l'addestramento.

Abbiamo dato a ogni fotografia un peso pari a 1/√(lunghezza dell'episodio). Risultato:
- migliore in **4 prove su 4**;
- validazione **da 0,647 a 0,661**.

### Idea 2: i blocchi, e cosa succede dopo (la svolta)

Le fotografie etichettate arrivano a **blocchi**: istanti consecutivi a 3 secondi l'uno dall'altro, mentre il controllore lavora su qualcosa. In addestramento ci sono 3.891 blocchi, con una mediana di 3 fotografie ciascuno.

- Dentro un blocco la risposta resta la stessa il **95%** delle volte.
- L'**87%** dei cambi di compito avviene tra un blocco e l'altro.

*Schema illustrativo: gli orari sono di esempio.*

```
tempo →    10:00:00   10:00:03   10:00:06   10:00:09    ...pausa...    10:00:52
           [════════════ blocco ════════════]                          [blocco successivo
compito X   in lista   in lista   in lista   in lista                   NON più in lista ← finito!
compito Y   in lista   in lista   in lista   in lista                   in lista
```

**L'osservazione chiave.** Quando il controllore completa un compito, ad esempio prende in carico un aereo, AWARE lo toglie dalla lista. Quindi il compito su cui lavorava tende a **sparire proprio alla fine del blocco**.

| Indizio (validazione) | Compito corretto | Altri compiti |
|---|---:|---:|
| Il compito viene visto per l'ultima volta esattamente a fine blocco | **49%** | 3% |
| L'aereo riceve un'istruzione dal controllore durante il blocco | **39%** | 4% |

**Come lo usiamo.** Il file di test contiene le **sessioni intere**, quindi per ogni istante possiamo leggere gli istanti successivi della stessa sessione. Non usiamo mai le risposte, solo orari, nomi dei compiti e tempi delle istruzioni. Abbiamo costruito indizi come:
- "quanto manca alla fine del blocco";
- "quando il compito sparisce rispetto alla fine del blocco";
- "quando arriva la prossima istruzione su questo aereo".

**Risultato:** la CV passa **da 0,661 a 0,901** e la validazione **da 0,661 a 0,929**, e il risultato regge su tutte e 4 le prove (da 0,875 a 0,938).

L'ultimo ritocco aggiunge lo stato dell'aereo **subito prima e subito dopo** il blocco: coordinamento, livello autorizzato, presa in carico e così via. Il modello finale è la **media** tra il modello con e quello senza questi indizi. Così si ottiene CV 0,906 e validazione 0,930, ed è la versione stabile su entrambe le misure.

> ⚠ **Nota di trasparenza.** Il modello finale usa informazioni *successive* all'istante da prevedere, presenti nel file di test. È corretto per la competizione, ma **non è un sistema in tempo reale**: in sala controllo il "dopo" non è ancora disponibile. Il miglior modello che usa solo il passato ottiene 0,661.

---

## 5. Cosa NON ha funzionato (e cosa ci ha insegnato)

Ogni prova è stata valutata con le stesse regole, e la maggior parte delle idee "ovvie" **peggiorava** il risultato su controllori nuovi.

| Idea | Effetto in CV | Lezione |
|---|---|---|
| Classifiche e confronti tra compiti della stessa lista | −0,012, peggiore in 4 prove su 4 | Le colonne `l_` fornite già contengono questa informazione; aggiungerne altre porta il modello a imparare abitudini dei singoli controllori |
| Statistiche per aereo, differenze con l'istante precedente | da −0,002 a −0,009 | Rumore più che segnale |
| "Memoria" dei punteggi negli istanti passati | nessun guadagno | Le colonne esistenti sono già stabili nel tempo |
| Memorizzare quale compito viene scelto in ogni scenario | −0,005 | Controllori diversi decidono diversamente |
| Altri algoritmi (XGBoost, CatBoost), altre impostazioni | nessuno meglio di LightGBM | — |

**La GPU** (NVIDIA MX450):
- **XGBoost** è circa **20 volte più veloce** su GPU (misurato con la CPU già occupata da un altro calcolo).
- **LightGBM** è più veloce sulla CPU a 8 core.
- **CatBoost** resta troppo lento anche su GPU.

La GPU ha accelerato gli esperimenti, ma non ha migliorato l'accuratezza.

---

## 6. Dove sbaglia ancora il modello

Punteggio top-3 per tipo di compito (CV, modello con informazioni sui blocchi):

| Tipo di compito | Top-3 |
|---|---:|
| Transfer (passaggio al settore successivo) | 0,98 |
| Assume (presa in carico) | 0,98 |
| Zone conflict (zona militare) | 0,92 |
| Return to route (rientro in rotta) | 0,84 |
| Entry coordination | 0,71 |
| **Exit coordination** | **0,57** |

Gli errori rimasti riguardano soprattutto i compiti che **non spariscono** quando il controllore ci lavora, come i coordinamenti, che restano in lista.

Il **35%** dei compiti corretti resta in lista per oltre 30 secondi dopo la fine del blocco:
- su questi casi il modello indovina il **78%**;
- su tutti gli altri il **96,5%**.

---

## 7. Prossimi passi

1. **Abbinare il tipo di istruzione al tipo di compito.** Un cambio di livello indica uscita o coordinamento; un cambio di rotta indica zona militare o rientro in rotta.
2. **Usare i blocchi vicini.** Il controllore spesso torna sullo stesso compito non finito nel blocco successivo.
3. **Ragionare per blocco intero** invece che per singola fotografia.

---

## Appendice: riprodurre il risultato

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
PY=.venv/bin/python
$PY -m src.validate_data                                   # ① controllo dati
$PY -m tests.test_metric_vs_scorer                         # ② metrica = scorer.py
$PY -m src.train baselines                                 # ④ baseline
$PY -m src.train cv lr_rw5_lead_3s lr_rw5_leadchg_3s       # ⑤⑥⑦ CV + validazione
$PY -m src.blend_fixed lr_rw5_lead_3s lr_rw5_leadchg_3s    # ⑦ media dei due modelli
$PY -m src.train final lr_rw5_lead_3s                      # ⑧ riaddestramento finale
$PY -m src.train final lr_rw5_leadchg_3s
$PY -m src.predict lr_rw5_lead_3s,lr_rw5_leadchg_3s --out submissions/submission.csv   # ⑨
$PY -m src.validate_submission submissions/submission.csv
```

| File | Contenuto |
|---|---|
| `reports/REPORT.md` | report tecnico completo |
| `reports/experiments.csv` | tutti gli esperimenti, compresi quelli scartati |
| `reports/comparisons.csv` | confronti a coppie tra modelli |
| `reports/error_analysis.md` | dove sbaglia il modello |
| `src/features.py` | costruzione degli indizi (gruppi `lead_blk`, `lead_chg` per i blocchi) |
| `src/train.py` | addestramento e selezione |
| `src/predict.py` | generazione della submission |
