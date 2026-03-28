# Analisi comparativa delle prestazioni di modelli di deep learning su dati sintetici e rumorosi

Tesi di Laurea Magistrale in Informatica – Università degli Studi di Milano-Bicocca  
**Candidato:** Daniel Satriano | **Relatore:** Prof. Andrea Maurino | **A.A. 2024/2025**

---

## Descrizione

Questo repository contiene il codice, i notebook e i sorgenti LaTeX utilizzati per la tesi magistrale che analizza l'impatto di diverse tipologie di rumore sui dati (missing values, rumore sulle feature, outlier, duplicati, rumore sulle etichette) sulle performance di modelli di deep learning (MLP e LSTM), sfruttando la libreria **PuckTrick** per la generazione controllata del rumore su backend Pandas e Apache Spark.

Il dataset utilizzato è il **Metro Interstate Traffic Volume dataset**, analizzato e corrotto sperimentalmente a vari livelli di rumore (10%, 20%, 30%, 50%).

---

## Struttura della repository

```
Master-thesis/
│
├── main.tex                         # Entry point LaTeX della tesi
├── frontispiece.tex                 # Frontespizio
├── Acronyms.tex                     # Lista degli acronimi
├── bibliography.bib                 # Bibliografia
├── Requirements.txt                 # Dipendenze Python
├── Tesi.pdf                         # PDF della tesi compilata
│
├── chapters/                        # Capitoli della tesi in LaTeX
├── images/                          # Immagini e grafici usati nella tesi
│
├── notebook/
│   ├── Metro-dt-analysis.ipynb      # Analisi esplorativa del dataset
│   ├── pucktrick_noiseDT_analisys.py # Analisi del dataset dopo la corruzione con PuckTrick
│   │
│   ├── MLP/                         # Esperimenti con Multilayer Perceptron
│   │
│   └── LSTM/                        # Esperimenti con Long Short-Term Memory
│       ├── lstm-tuning.py           # Tuning degli iperparametri LSTM
│       ├── lstm_datasets_export.py  # Generazione ed esportazione dei dataset corrotti
│       ├── lstm_model_only_train.py # Training del modello LSTM
│       ├── lstm_tuning_images.py    # Generazione immagini per il tuning
│       ├── lstm_results_final_base.jsonl  # Risultati sperimentali (JSONL)
│       ├── lstm_analisi_risultati.ipynb   # Analisi e visualizzazione dei risultati LSTM
│       └── new_experiments/         # Esperimenti aggiuntivi LSTM
│
└── pucktrick_conversion_results/    # Output del benchmark Pandas vs Spark
```

---

## Tecnologie utilizzate

- **Python 3.x** — linguaggio principale
- **PyTorch** — implementazione di MLP e LSTM
- **Apache Spark** (MLlib, TorchDistributor) — backend distribuito per PuckTrick e training su cluster
- **PuckTrick** — libreria per l'introduzione controllata di rumore su DataFrame (Pandas/Spark)
- **pandas**, **numpy**, **scikit-learn** — preprocessing e analisi dati
- **LaTeX** — redazione del documento di tesi

---

## Installazione

```bash
git clone https://github.com/Code-Dani/Master-thesis.git
cd Master-thesis

python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

pip install -r Requirements.txt
```

---

## Flusso degli esperimenti

1. **Analisi esplorativa** → `notebook/Metro-dt-analysis.ipynb`
2. **Corruzione del dataset con PuckTrick** → `notebook/pucktrick_noiseDT_analisys.py`
3. **Generazione dataset LSTM** → `notebook/LSTM/lstm_datasets_export.py`
4. **Tuning iperparametri** → `notebook/LSTM/lstm-tuning.py` / equivalente in `notebook/MLP/`
5. **Training e valutazione** → `notebook/LSTM/lstm_model_only_train.py` / equivalente in `notebook/MLP/`
6. **Analisi risultati** → `notebook/LSTM/lstm_analisi_risultati.ipynb`

---

## Citazione

Se utilizzi questo codice o ti basi su questo lavoro, cita:

> Daniel Satriano, *Analisi comparativa delle prestazioni di modelli di deep learning su dati sintetici e rumorosi*, Tesi di Laurea Magistrale, Università degli Studi di Milano-Bicocca, A.A. 2024/2025.
