# Analisi comparativa delle prestazioni di modelli di deep learning su dati sintetici e rumorosi

Repository a supporto della tesi di laurea *“Analisi comparativa delle prestazioni di modelli di deep learning su dati sintetici e rumorosi”*, Corso di Laurea in Informatica – Università degli Studi di Milano-Bicocca, A.A. 2024/2025. [file:1]

## Obiettivo del progetto

Lo scopo della repo è fornire codice, script e configurazioni utilizzati per:
- Generare e corrompere dataset sintetici tramite diversi tipi di rumore (missing, noise, outliers, duplicated, rumore sulle labels). [file:1]
- Addestrare e valutare modelli di deep learning (MLP e LSTM) su dati puliti e rumorosi. [file:1]
- Analizzare l’impatto del rumore sulle performance (F1-score, AUC-ROC) e sull’affidabilità dei modelli. [file:1]

## Contenuti della repository

La struttura reale può variare; di seguito uno schema consigliato:

- `data/`
  - Dataset originali e versioni corrotte (o script per scaricarli/generarli).
- `notebooks/`
  - Notebook per esplorazione, analisi descrittiva e visualizzazione dei risultati.
- `src/`
  - `pucktrick_backend/`: implementazione o integrazione della libreria PuckTrick per introdurre rumore, con supporto sia Pandas che Spark. [file:1]
  - `experiments_mlp/`: script per addestramento, tuning e valutazione del MLP. [file:1]
  - `experiments_lstm/`: script per addestramento, tuning e valutazione del LSTM. [file:1]
  - `utils/`: funzioni di supporto (preprocessing, metriche, logging, gestione esperimenti).
- `configs/`
  - File di configurazione (es. YAML/JSON) per definire parametri di esperimenti, set di feature, percentuali di rumore, ecc. [file:1]
- `results/`
  - Tabelle e log con F1-score e AUC-ROC per i vari scenari di rumore, in linea con le tabelle di riferimento riportate in appendice alla tesi. [file:1]
- `environment.yml` / `requirements.txt`
  - Dipendenze Python per riprodurre gli esperimenti.
- `README.md`
  - Questo file.

Adatta i nomi delle cartelle ai nomi effettivamente presenti nella tua repo.

## Tecnologie e dipendenze

Principali strumenti utilizzati nel progetto (da rifinire in base alla repo reale):

- **Linguaggio**: Python 3.x
- **Librerie scientifiche**: `numpy`, `pandas`, `scikit-learn`, `matplotlib` / `seaborn`.
- **Deep learning**: `PyTorch` (per l’implementazione di MLP e LSTM). [file:1]
- **Big data e distribuito**: `Apache Spark` (MLlib, TorchDistributor) per la gestione di dataset distribuiti e l’esecuzione degli esperimenti su cluster. [file:1]
- **Gestione esperimenti**: eventuale uso di strumenti per logging/track delle run (es. `mlflow`, `tensorboard`), se presenti nella repo.

Esempio di installazione:

```bash
# con virtualenv
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

Oppure:

```bash
# con Conda
conda env create -f environment.yml
conda activate tesi-noise-env
```

## Riproduzione degli esperimenti

Di seguito un flusso generico per replicare i risultati della tesi (personalizza i nomi degli script/notebook in base alla repo):

1. **Preparazione dati**
   - Posiziona i dataset grezzi in `data/` oppure esegui lo script/notebook di generazione:  
     ```bash
     python src/data/make_dataset.py
     ```
   - (Opzionale) Esegui uno script di analisi esplorativa:  
     ```bash
     jupyter notebook notebooks/EDA.ipynb
     ```

2. **Introduzione del rumore con PuckTrick**
   - Esegui gli script per applicare le diverse tipologie di rumore:
     ```bash
     python src/pucktrick_backend/apply_missing.py
     python src/pucktrick_backend/apply_noise.py
     python src/pucktrick_backend/apply_outliers.py
     python src/pucktrick_backend/apply_duplicated.py
     python src/pucktrick_backend/apply_labels_noise.py
     ```
   - Gli script dovrebbero produrre i dataset corrotti nelle relative sottocartelle di `data/`. [file:1]

3. **Esperimenti con MLP**
   - Tuning degli iperparametri su dato pulito:  
     ```bash
     python src/experiments_mlp/tuning_clean.py
     ```
   - Esecuzione su dataset rumorosi (per ciascun tipo e percentuale di rumore):  
     ```bash
     python src/experiments_mlp/run_experiments.py --noise-type missing --noise-level 0.1
     python src/experiments_mlp/run_experiments.py --noise-type noise --noise-level 0.2
     # ...
     ```
   - I risultati (F1-score, AUC-ROC, intervalli di confidenza al 95%) vengono salvati in `results/mlp/`. [file:1]

4. **Esperimenti con LSTM**
   - Tuning iperparametri su dato pulito:  
     ```bash
     python src/experiments_lstm/tuning_clean.py
     ```
   - Esecuzione del modello su dataset rumorosi:  
     ```bash
     python src/experiments_lstm/run_experiments.py --noise-type duplicated --noise-level 0.3
     # ...
     ```
   - I risultati sono salvati in `results/lstm/`, con tabelle analoghe a quelle riportate in appendice B della tesi. [file:1]

5. **Analisi dei risultati**
   - Lancia i notebook di analisi per generare grafici e tabelle (es. effetto del rumore su F1 e AUC per ogni feature e modello):  
     ```bash
     jupyter notebook notebooks/analysis_mlp.ipynb
     jupyter notebook notebooks/analysis_lstm.ipynb
     ```

## Principali esperimenti inclusi

Gli esperimenti della tesi coprono, in sintesi: [file:1]

- Valutazione di un MLP su:
  - Dataset pulito (baseline).
  - Dataset corrotti con:
    - Rumore sulle feature (missing, noise, outliers) per diverse feature (DV_pressure, TP3, Oil_temperature) e diversi livelli di rumore (10%, 20%, 30%, 50%). [file:1]
    - Rumore sulle etichette (labels) e duplicazione di righe (duplicated), anche mirata sulla classe minoritaria. [file:1]
    - Corruzione simultanea di più feature (es. TP3 e Reservoirs). [file:1]

- Valutazione di un LSTM su:
  - Dataset pulito (baseline su 20 esecuzioni). [file:1]
  - Dataset con rumore da duplicated, labels e altre forme di perturbazione, monitorando l’effetto su F1-score e AUC-ROC. [file:1]

I risultati numerici dettagliati sono riportati nelle tabelle di riferimento delle appendici A e B del documento. [file:1]

## Citazione

Se utilizzi questo codice o ti basi su questi esperimenti per il tuo lavoro, ti prego di citare la seguente tesi:

> Daniel Satriano, *Analisi comparativa delle prestazioni di modelli di deep learning su dati sintetici e rumorosi*, Tesi di Laurea, Università degli Studi di Milano-Bicocca, A.A. 2024/2025. [file:1]

---

Se mi indichi la struttura effettiva della repo (cartelle e nomi file principali), posso adattare il README in modo perfettamente aderente al tuo progetto.

```xml
```
