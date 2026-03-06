"""
lstm-tuning.py
==============
Ricerca degli iperparametri LSTM sul dato pulito.

Esecuzione: python lstm-tuning.py

Strategia
---------
- Campione del 20% del training set, ordinato per timestamp (no data leakage)
- Split interno 80/20 TEMPORALE — il test set non viene mai toccato
- Grid search manuale (no CrossValidator: non supporta PyTorch)
- Training su driver — il subset è abbastanza piccolo (~170k righe)
- Salva i best params in PARAMS_PATH → da copiare manualmente in lstm-esperimenti.py

Output
------
  /home/PuckTrickadmin/RESULTS/lstm_best_params.json
"""

import time
import json
from itertools import product

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

from pucktrick import PuckTrick, Engine
from sklearn.metrics import f1_score, roc_auc_score

# ── Costanti ───────────────────────────────────────────────────────────────
ALL_FEATURES = [
    "TP2_scaled", "TP3_scaled", "H1_scaled", "DV_pressure_scaled",
    "Reservoirs_scaled", "Oil_temperature_scaled", "Motor_current_scaled",
    "COMP", "DV_eletric", "Towers", "MPG", "LPS", "Pressure_switch"
]
NOISE_FEATURES = ["DV_pressure_scaled", "Oil_temperature_scaled", "TP3_scaled"]
LABEL_COL      = "target"
N_FEATURES     = len(ALL_FEATURES)   # 13

PARAMS_PATH = "/home/PuckTrickadmin/RESULTS/lstm_best_params.json"

# ── Griglia iperparametri ──────────────────────────────────────────────────
#
# window_size — timestep a 1 Hz (secondi di contesto per ogni finestra)
#   Il dataset è un compressore pneumatico: i guasti hanno due nature distinte:
#   - Anomalie rapide (spike pressione/corrente):     finestre corte  30-60s
#   - Anomalie lente (degrado termico, perdita pressione progressiva): 120-300s
#   Testare entrambi gli estremi è necessario per non precludere nessun pattern.
#   window_size=10 escluso: troppo poco contesto per pattern meccanici con 13 feature.
#   window_size=300 riduce il pool di finestre di 299 per sequenza — trascurabile
#   su 856k righe, il costo reale è solo nel tempo di training (+67 combinazioni).
#
# fc_architecture — specifica LSTM hidden size + FC nascosti prima dell'output.
#   Convenzione: fc_architecture[0] = hidden_size dello strato LSTM
#                fc_architecture[1:] = dimensioni dei layer FC intermedi
#                L'output layer FC(2) viene aggiunto automaticamente dal modello.
#
#   Esempi (parallelo diretto con MLP layers):
#     [64]          → LSTM(64)  → FC(2)              come MLP [13, 64, 2]
#     [128]         → LSTM(128) → FC(2)              come MLP [13, 128, 2]
#     [128, 64]     → LSTM(128) → FC(64) → FC(2)     come MLP [13, 128, 64, 2]
#     [256, 128]    → LSTM(256) → FC(128) → FC(2)    come MLP [13, 256, 128, 2]
#     [256, 128, 64]→ LSTM(256) → FC(128)→ FC(64)→ FC(2)
#
#   Con 13 feature in input, LSTM hidden < 64 comprime troppo la rappresentazione
#   temporale. 256+ esplora capacità per pattern complessi ma rischia overfitting
#   sulla classe guasto (1.29% del training set).
#
# lstm_layers — strati LSTM impilati (stacked LSTM)
#   1 → baseline; sufficiente per la maggior parte dei task di anomaly detection
#   2 → gerarchia temporale astratta; raddoppia parametri e tempo di training
#
# lr — learning rate ottimizzatore Adam
#   1e-3 → standard Adam; punto di partenza consolidato
#   1e-4 → più stabile con loss pesata su dataset sbilanciato
#
# num_epochs — sul subset di tuning (~170k righe, batch=512)
#   5  → rapido, stima approssimativa della convergenza
#   10 → sufficiente per convergenza stabile sul subset
#   (20 epoch riservate al loop sperimentale sul dataset completo)
#
# TOTALE: 4 × 5 × 2 × 2 × 2 = 160 combinazioni
# Stima tempo su driver CPU: ~5-8h

GRID = {
    "window_size":    [30, 60, 120, 300],
    "fc_architecture": [
        [64],
        [128],
        [256],
        [128, 64],
        [256, 128],
    ],
    "lstm_layers":    [1, 2],
    "lr":             [1e-3, 1e-4],
    "num_epochs":     [5, 10],
}

BATCH_SIZE = 512  # fisso — buon compromesso throughput/memoria


# =============================================================================
# MODELLO
# =============================================================================

class LSTMClassifier(nn.Module):
    """
    LSTM binario per anomaly detection su serie temporale con testa FC configurabile.

    Input  : (batch, window_size, n_features=13)
    Output : (batch, 2)  — logit per classe 0 (normale) e 1 (guasto)

    Parametri
    ---------
    input_size     : numero di feature in ingresso (13)
    fc_architecture: lista che specifica hidden_size LSTM + FC nascosti.
                     fc_architecture[0]  → hidden_size dello strato LSTM
                     fc_architecture[1:] → dimensioni dei FC layer intermedi
                     Il layer di output FC(2) è sempre aggiunto in fondo.

                     Esempi (parallelo con layers MLP):
                       [64]           → LSTM(64) → FC(2)
                       [128]          → LSTM(128) → FC(2)
                       [128, 64]      → LSTM(128) → FC(64) → ReLU → FC(2)
                       [256, 128]     → LSTM(256) → FC(128) → ReLU → FC(2)
                       [256, 128, 64] → LSTM(256) → FC(128) → ReLU → FC(64) → ReLU → FC(2)

    lstm_layers    : numero di strati LSTM impilati (1 o 2)
    dropout        : applicato allo stacked LSTM solo se lstm_layers > 1
    """

    def __init__(self, input_size: int, fc_architecture: list, lstm_layers: int,
                 dropout: float = 0.2):
        super().__init__()
        hidden_size = fc_architecture[0]

        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = lstm_layers,
            batch_first = True,
            dropout     = dropout if lstm_layers > 1 else 0.0,
        )

        # Costruisce la testa FC: hidden_size → [dims intermedi] → 2
        fc_layers = []
        in_dim = hidden_size
        for out_dim in fc_architecture[1:]:
            fc_layers.append(nn.Linear(in_dim, out_dim))
            fc_layers.append(nn.ReLU())
            in_dim = out_dim
        fc_layers.append(nn.Linear(in_dim, 2))
        self.fc_head = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)           # (B, W, H)
        return self.fc_head(out[:, -1]) # (B, 2) — ultimo timestep


# =============================================================================
# DATASET — WINDOWING TEMPORALE
# =============================================================================

class TimeSeriesWindowDataset(Dataset):
    """
    Finestre temporali sovrapposte (stride=1) su un array ordinato per timestamp.

    X[i] = features[i : i + window_size]     shape (window_size, n_features)
    y[i] = labels[i + window_size - 1]       etichetta dell'ultimo step della finestra

    Nota: il sorting per timestamp DEVE avvenire PRIMA di passare gli array
    a questa classe — l'ordine degli elementi è critico per la correttezza
    della serie temporale.
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray, window_size: int):
        self.features    = torch.tensor(features, dtype=torch.float32)
        self.labels      = torch.tensor(labels,   dtype=torch.long)
        self.window_size = window_size

    def __len__(self) -> int:
        return len(self.features) - self.window_size + 1

    def __getitem__(self, idx: int):
        return (
            self.features[idx : idx + self.window_size],   # (W, F)
            self.labels[idx + self.window_size - 1],       # scalare
        )


# =============================================================================
# UTILS
# =============================================================================

def compute_class_weight(labels: np.ndarray) -> torch.Tensor:
    """
    Peso inversamente proporzionale alla frequenza di classe.
    w_neg = 1.0,  w_pos = n_neg / n_pos
    Compensa lo sbilanciamento ~98% normale / ~2% guasto.
    """
    n_pos = max((labels == 1).sum(), 1)
    n_neg = (labels == 0).sum()
    return torch.tensor([1.0, n_neg / n_pos], dtype=torch.float32)


def train_and_evaluate(
    train_features:  np.ndarray,
    train_labels:    np.ndarray,
    val_features:    np.ndarray,
    val_labels:      np.ndarray,
    window_size:     int,
    fc_architecture: list,
    lstm_layers:     int,
    lr:              float,
    num_epochs:      int,
    batch_size:      int  = BATCH_SIZE,
    seed:            int  = 42,
    device:          str  = "cpu",
) -> dict:
    """Addestra LSTMClassifier e restituisce F1 weighted e AUC sul set di validazione."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = TimeSeriesWindowDataset(train_features, train_labels, window_size)
    val_ds   = TimeSeriesWindowDataset(val_features,   val_labels,   window_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    model     = LSTMClassifier(train_features.shape[1], fc_architecture, lstm_layers).to(device)
    weight    = compute_class_weight(train_labels).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(num_epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

    model.eval()
    preds, probs, labels_out = [], [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            logits = model(xb.to(device))
            probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            preds.extend(logits.argmax(dim=1).cpu().numpy())
            labels_out.extend(yb.numpy())

    labels_out = np.array(labels_out)
    preds      = np.array(preds)
    probs      = np.array(probs)

    f1  = f1_score(labels_out, preds, average="weighted", zero_division=0)
    try:
        auc = roc_auc_score(labels_out, probs)
    except ValueError:
        auc = 0.0

    return {"f1": f1, "auc": auc}


# =============================================================================
# INIT SPARK
# =============================================================================

def init_spark():
    MASTER_URL  = "spark://10.0.1.8:7077"
    DRIVER_HOST = "10.0.1.8"

    spark = SparkSession.builder \
        .appName("MetroPT_LSTM_Tuning") \
        .master(MASTER_URL) \
        .config("spark.submit.deployMode",      "client") \
        .config("spark.executor.instances",     "4") \
        .config("spark.executor.cores",         "4") \
        .config("spark.executor.memory",        "13g") \
        .config("spark.driver.memory",          "8g") \
        .config("spark.driver.host",            DRIVER_HOST) \
        .config("spark.driver.bindAddress",     DRIVER_HOST) \
        .config("spark.sql.shuffle.partitions", "32") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    print("SparkSession creata — versione:", spark.version)

    DATA_PATH = "/home/PuckTrickadmin/DATASETS/MetroDT_Modified.parquet"
    raw_pdf   = pd.read_parquet(DATA_PATH)
    raw_df    = spark.createDataFrame(raw_pdf)
    del raw_pdf

    df = raw_df.select(
        F.col("timestamp"),
        *[F.col(c).cast(DoubleType()) for c in ALL_FEATURES],
        F.col("target").cast(DoubleType()),
    ).dropna()

    SPLIT_DATE    = "2020-06-01 00:00:00"
    pt            = PuckTrick(df, engine=Engine.SPARK)
    base_df       = pt.original

    base_train_df = base_df.filter(F.col("timestamp") < SPLIT_DATE)
    base_test_df  = base_df.filter(F.col("timestamp") >= SPLIT_DATE)

    base_train_df.cache()
    base_test_df.cache()

    n_train       = base_train_df.count()
    n_test        = base_test_df.count()
    n_train_fault = base_train_df.filter(F.col("target") == 1).count()
    n_test_fault  = base_test_df.filter(F.col("target") == 1).count()

    print(f"Training : {n_train:,} righe ({n_train_fault:,} guasti, {n_train_fault/n_train*100:.2f}%)")
    print(f"Test     : {n_test:,} righe ({n_test_fault:,} guasti, {n_test_fault/n_test*100:.2f}%)")
    print(f"Split    : {n_train/(n_train+n_test)*100:.1f}% train / {n_test/(n_train+n_test)*100:.1f}% test")

    return spark, base_train_df, base_test_df


# =============================================================================
# TUNING
# =============================================================================

def calcolo_iperparametri(base_train_df, base_test_df):
    """
    Grid search manuale su 120 combinazioni.

    Scelta implementativa — perché il tuning avviene sul training set
    ------------------------------------------------------------------
    Il training set viene usato sia per il tuning degli iperparametri sia
    per il loop sperimentale. Questo è corretto e intenzionale:

    - Il test set deve rimanere completamente isolato fino alla valutazione
      finale. Usarlo per il tuning introdurrebbe data leakage, invalidando
      tutti i confronti tra le 2.000 run sperimentali.
    - Il tuning richiede un segnale di validazione interno per confrontare
      le 120 combinazioni di iperparametri. Questo segnale viene ricavato
      con uno split temporale 55/45 INTERNO al training set.
    - Il loop sperimentale usa poi il 100% del training set (con rumore
      iniettato) e valuta sempre sul test set pulito — esattamente come
      per il modello MLP.

    La struttura è quindi:

        Training set (856k righe)
        ├── Tuning:       55% train interno → 45% val interna
        │                 (confronto tra le 120 combinazioni → best params)
        └── Esperimenti:  100% training (rumoroso) → valuta su test set

        Test set (660k righe) — mai toccato fino alla valutazione finale

    Split basato sulle date reali dei guasti
    ----------------------------------------
    Anziché usare uno split percentuale, il punto di taglio è fissato al
    2020-05-01 — tra i due guasti presenti nel training set:
      Guasto 1: 18/04/2020        → finisce in tune-train
      Guasto 2: 29-30/05/2020     → finisce in tune-val
    Questo garantisce un fault per parte indipendentemente dalla distribuzione
    temporale, senza nessuna euristica. Qualsiasi split percentuale fisso
    rischia di concentrare tutti i guasti in una sola parte (come verificato
    empiricamente: con split 55/45 tune-train aveva 0 guasti su 471k righe).

    Training su driver CPU (no TorchDistributor — i dati stanno in RAM).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # ── Split basato sulle date reali dei guasti ──────────────────────────
    # Il training set (feb-mag 2020) contiene esattamente due episodi di guasto:
    #   Guasto 1: 18/04/2020 — interamente nel periodo feb-apr
    #   Guasto 2: 29/05/2020 23:30 → 30/05/2020 06:00 — in maggio
    #
    # Splittando al 2020-05-01 si garantisce un fault per parte:
    #   Tune-train (feb → apr): contiene guasto 1  ✓
    #   Tune-val   (mag → giu): contiene guasto 2  ✓
    #
    # Questo è più solido di qualsiasi split percentuale perché si basa sulla
    # distribuzione reale dei guasti nel dataset — nessuna euristica necessaria.
    TUNE_SPLIT_DATE = "2020-05-01 00:00:00"

    tune_train_pdf = (
        base_train_df
        .filter(F.col("timestamp") < TUNE_SPLIT_DATE)
        .orderBy("timestamp")
        .select(ALL_FEATURES + [LABEL_COL])
        .toPandas()
    )
    tune_val_pdf = (
        base_train_df
        .filter(F.col("timestamp") >= TUNE_SPLIT_DATE)
        .orderBy("timestamp")
        .select(ALL_FEATURES + [LABEL_COL])
        .toPandas()
    )

    tr_f  = tune_train_pdf[ALL_FEATURES].values.astype(np.float32)
    tr_l  = tune_train_pdf[LABEL_COL].values.astype(np.int64)
    va_f  = tune_val_pdf[ALL_FEATURES].values.astype(np.float32)
    va_l  = tune_val_pdf[LABEL_COL].values.astype(np.int64)

    n_fault_tr = (tr_l == 1).sum()
    n_fault_va = (va_l == 1).sum()
    print(f"Tune-train : {len(tr_f):,} righe  ({n_fault_tr:,} guasti, {n_fault_tr/len(tr_f)*100:.2f}%)")
    print(f"Tune-val   : {len(va_f):,} righe  ({n_fault_va:,} guasti, {n_fault_va/len(va_f)*100:.2f}%)")

    # ── Grid search ────────────────────────────────────────────────────────
    keys   = list(GRID.keys())
    # Escludi la combinazione [256,128] + lstm_layers=2 — troppi parametri
    # per un dataset con 1.29% di fault, rischio overfitting sistematico.
    combos = [
        p for p in (dict(zip(keys, combo)) for combo in product(*GRID.values()))
        if not (p["fc_architecture"] == [256, 128] and p["lstm_layers"] == 2)
    ]
    print(f"Combinazioni da testare: {len(combos)}\n")

    results = []
    for i, p in enumerate(combos):  # p è già un dizionario
        t0 = time.time()

        m = train_and_evaluate(
            train_features  = tr_f,
            train_labels    = tr_l,
            val_features    = va_f,
            val_labels      = va_l,
            window_size     = p["window_size"],
            fc_architecture = p["fc_architecture"],
            lstm_layers     = p["lstm_layers"],
            lr              = p["lr"],
            num_epochs      = p["num_epochs"],
            batch_size      = BATCH_SIZE,
            seed            = 42,
            device          = device,
        )
        elapsed = time.time() - t0
        p.update({"f1": m["f1"], "auc": m["auc"], "elapsed_s": elapsed})
        results.append(p)

        arch_str = str(p["fc_architecture"])
        print(
            f"[{i+1:02d}/{len(combos)}]"
            f"  win={p['window_size']:2d}"
            f"  arch={arch_str:<16}"
            f"  lstm_lay={p['lstm_layers']}"
            f"  lr={p['lr']:.0e}"
            f"  ep={p['num_epochs']:2d}"
            f"  →  F1={m['f1']:.4f}  AUC={m['auc']:.4f}"
            f"  ({elapsed:.1f}s)"
        )

    # ── Best params ────────────────────────────────────────────────────────
    best = max(results, key=lambda r: r["f1"])
    print("\n" + "="*60)
    print("MIGLIORI IPERPARAMETRI")
    print("="*60)
    for k in ["window_size", "fc_architecture", "lstm_layers", "lr", "num_epochs"]:
        print(f"  {k:<18}: {best[k]}")
    print(f"  {'batch_size':<15}: {BATCH_SIZE}  (fisso)")
    print(f"  {'F1 tuning':<15}: {best['f1']:.4f}")

    # ── Baseline sul test set completo con i best params ───────────────────
    print("\nCalcolo baseline su test set pulito (training completo)...")

    train_pdf = (
        base_train_df
        .orderBy("timestamp")
        .select(ALL_FEATURES + [LABEL_COL])
        .toPandas()
    )
    test_pdf = (
        base_test_df
        .orderBy("timestamp")
        .select(ALL_FEATURES + [LABEL_COL])
        .toPandas()
    )

    baseline = train_and_evaluate(
        train_features  = train_pdf[ALL_FEATURES].values.astype(np.float32),
        train_labels    = train_pdf[LABEL_COL].values.astype(np.int64),
        val_features    = test_pdf[ALL_FEATURES].values.astype(np.float32),
        val_labels      = test_pdf[LABEL_COL].values.astype(np.int64),
        window_size     = best["window_size"],
        fc_architecture = best["fc_architecture"],
        lstm_layers     = best["lstm_layers"],
        lr              = best["lr"],
        num_epochs      = best["num_epochs"],
        batch_size      = BATCH_SIZE,
        seed            = 42,
        device          = device,
    )

    print(f"\nBASELINE (0% rumore)")
    print(f"  F1  : {baseline['f1']:.4f}")
    print(f"  AUC : {baseline['auc']:.4f}")

    # ── Salvataggio ────────────────────────────────────────────────────────
    output = {
        "window_size":    best["window_size"],
        "fc_architecture": best["fc_architecture"],
        "lstm_layers":    best["lstm_layers"],
        "lr":             best["lr"],
        "num_epochs":     best["num_epochs"],
        "batch_size":     BATCH_SIZE,
        "baseline_f1":    baseline["f1"],
        "baseline_auc":   baseline["auc"],
        "all_results":    results,
    }
    with open(PARAMS_PATH, "w") as fh:
        json.dump(output, fh, indent=2)

    print(f"\nParametri salvati in: {PARAMS_PATH}")
    print("\n>>> Copia i valori BEST_* in lstm-esperimenti.py prima di avviare il loop.\n")

    return output


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    spark, base_train_df, base_test_df = init_spark()
    calcolo_iperparametri(base_train_df, base_test_df)
    spark.stop()