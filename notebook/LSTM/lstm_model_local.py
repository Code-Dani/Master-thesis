"""
lstm_model.py
=============
Loop sperimentale LSTM — 1.100 run totali.
PuckTrick Engine.SPARK per iniezione rumore (coerente con la tesi).
PyTorch single-process locale per il training.

Esecuzione: python lstm_model.py
"""

# DEVE stare prima di qualsiasi import Spark/PuckTrick
import os
import sys
os.environ["PYSPARK_PYTHON"]        = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ["SPARK_LOCAL_IP"]        = "127.0.0.1"

import time
import json
import signal
import traceback
from datetime import datetime

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, roc_auc_score

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pucktrick import PuckTrick, Engine

# =============================================================================
# ██ BEST PARAMS — config #6 da lstm-tuning.py ██
# =============================================================================

BEST_WINDOW_SIZE: int   = 30
BEST_HIDDEN_SIZE: int   = 64
BEST_NUM_LAYERS:  int   = 2
BEST_LR:          float = 1e-3
BEST_EPOCHS:      int   = 10
BEST_BATCH_SIZE:  int   = 2048

# =============================================================================

ALL_FEATURES = [
    "TP2_scaled", "TP3_scaled", "H1_scaled", "DV_pressure_scaled",
    "Reservoirs_scaled", "Oil_temperature_scaled", "Motor_current_scaled",
    "COMP", "DV_eletric", "Towers", "MPG", "LPS", "Pressure_switch",
]
NOISE_FEATURES   = ["DV_pressure_scaled", "Oil_temperature_scaled", "TP3_scaled"]
NOISE_NO_FEATURE = ["labels", "duplicated"]
LABEL_COL        = "target"
N_FEATURES       = len(ALL_FEATURES)   # 13

NOISE_TYPES  = ["duplicated", "labels", "missing", "noisy", "outliers"]
NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.5]
N_RUNS       = 20
T_VALUE_95   = 2.093

QUICK_MODE = True
_SEEDS     = range(5) if QUICK_MODE else range(N_RUNS)

RESULTS_PATH = "lstm_results.jsonl"
DATA_PATH    = r"D:\Users\satri\Pictures\github\Deep-Learning-Robustness-Study\notebook\Dataset\MetroDT_Modified.parquet"
SPLIT_DATE   = "2020-06-01 00:00:00"


# =============================================================================
# DATASET CON WINDOWING
# =============================================================================

class _WinDS(Dataset):
    def __init__(self, feats: np.ndarray, labs: np.ndarray, ws: int):
        self.f  = torch.tensor(feats, dtype=torch.float32)
        self.l  = torch.tensor(labs,  dtype=torch.long)
        self.ws = ws
    def __len__(self):
        return len(self.f) - self.ws + 1
    def __getitem__(self, i):
        return self.f[i : i + self.ws], self.l[i + self.ws - 1]


# =============================================================================
# MODELLO LSTM
# =============================================================================

class _LSTM(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = n_features,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = 0.2 if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 2)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])


# =============================================================================
# TRAINING LOCALE
# =============================================================================

def _train_local(
    train_features: np.ndarray,
    train_labels:   np.ndarray,
    test_features:  np.ndarray,
    test_labels:    np.ndarray,
    window_size:    int,
    hidden_size:    int,
    num_layers:     int,
    lr:             float,
    num_epochs:     int,
    batch_size:     int,
    seed:           int,
) -> dict:

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds     = _WinDS(train_features, train_labels, window_size)
    test_ds      = _WinDS(test_features,  test_labels,  window_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    model     = _LSTM(N_FEATURES, hidden_size, num_layers)
    n_pos     = max(int((train_labels == 1).sum()), 1)
    n_neg     = int((train_labels == 0).sum())
    weight    = torch.tensor([1.0, n_neg / n_pos], dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(num_epochs):
        for xb, yb in train_loader:
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

    model.eval()
    preds, probs, labs = [], [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb)
            probs.extend(torch.softmax(logits, dim=1)[:, 1].numpy())
            preds.extend(logits.argmax(dim=1).numpy())
            labs.extend(yb.numpy())

    labs_np  = np.array(labs)
    preds_np = np.array(preds)
    probs_np = np.array(probs)

    f1 = f1_score(labs_np, preds_np, average="weighted", zero_division=0)
    try:
        auc = roc_auc_score(labs_np, probs_np)
    except ValueError:
        auc = 0.0

    return {"f1": float(f1), "auc": float(auc)}


# =============================================================================
# INIT — Spark + PuckTrick Engine.SPARK
# =============================================================================

def init_data():
    spark = (
        SparkSession.builder
        .appName("MetroPT_LSTM_Experiments")
        .master("local[*]")
        .config("spark.driver.host",            "127.0.0.1")
        .config("spark.driver.bindAddress",     "127.0.0.1")
        .config("spark.driver.memory",          "8g")
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.python.worker.faulthandler.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    print("SparkSession creata — versione:", spark.version)

    raw_pdf = pd.read_parquet(DATA_PATH)
    raw_df  = spark.createDataFrame(raw_pdf)
    del raw_pdf

    df = raw_df.select(
        F.col("timestamp"),
        *[F.col(c).cast(DoubleType()) for c in ALL_FEATURES],
        F.col("target").cast(DoubleType()),
    )

    pt      = PuckTrick(df, engine=Engine.SPARK)
    base_df = pt.original

    base_train_df = base_df.filter(F.col("timestamp") <  SPLIT_DATE)
    base_test_df  = base_df.filter(F.col("timestamp") >= SPLIT_DATE)

    base_train_df.cache()
    base_test_df.cache()

    n_train       = base_train_df.count()
    n_test        = base_test_df.count()
    n_train_fault = base_train_df.filter(F.col(LABEL_COL) == 1).count()
    n_test_fault  = base_test_df.filter(F.col(LABEL_COL)  == 1).count()

    print(f"Training : {n_train:,} righe  ({n_train_fault:,} guasti, {n_train_fault/n_train*100:.2f}%)")
    print(f"Test     : {n_test:,}  righe  ({n_test_fault:,}  guasti, {n_test_fault/n_test*100:.2f}%)")
    print(f"Split    : {n_train/(n_train+n_test)*100:.1f}% train / {n_test/(n_train+n_test)*100:.1f}% test")
    print(f"QUICK_MODE: {QUICK_MODE} — seed: {list(_SEEDS)}")

    pt_train = PuckTrick(
        base_df.filter(F.col("timestamp") < SPLIT_DATE),
        engine=Engine.SPARK,
    )

    # Pre-calcola test arrays una volta sola
    test_pdf = (
        base_test_df
        .orderBy("timestamp")
        .select(ALL_FEATURES + [LABEL_COL])
        .toPandas()
    )
    test_features = np.array(test_pdf[ALL_FEATURES], dtype=np.float32)
    test_labels   = np.array(test_pdf[LABEL_COL],    dtype=np.int64)

    print(f"N_TRAIN_BASE: {n_train:,}")
    return spark, pt_train, test_features, test_labels, n_train


# =============================================================================
# SINGOLO RUN SPERIMENTALE
# =============================================================================

def run_experiment(
    pt_train,
    test_features: np.ndarray,
    test_labels:   np.ndarray,
    n_train_base:  int,
    noise_type:    str,
    feature_arg:   list,
    feature_label: str,
    percentage:    float,
    seed:          int,
) -> dict:
    t0 = time.time()

    # ── Iniezione rumore via Spark ─────────────────────────────────────
    if percentage == 0.0:
        noisy_df = pt_train.original
    elif noise_type in ("duplicated", "labels"):
        noisy_df = getattr(pt_train, noise_type)(
            percentage = percentage,
            columns    = [LABEL_COL],
            seed       = seed,
        )
    else:
        noisy_df = getattr(pt_train, noise_type)(
            percentage = percentage,
            columns    = feature_arg,
            seed       = seed,
        )

    noisy_df = noisy_df.orderBy("timestamp")
    noisy_df.cache()
    n_train  = noisy_df.count() if noise_type == "duplicated" else n_train_base

    # ── Raccolta su driver ─────────────────────────────────────────────
    train_pdf = noisy_df.select(ALL_FEATURES + [LABEL_COL]).toPandas()
    noisy_df.unpersist()

    # ── Mean imputer ──────────────────────────────────────────────────
    train_means = train_pdf[ALL_FEATURES].mean()
    train_pdf[ALL_FEATURES] = train_pdf[ALL_FEATURES].fillna(train_means)

    train_features = np.array(train_pdf[ALL_FEATURES], dtype=np.float32)
    train_labels   = np.array(train_pdf[LABEL_COL],    dtype=np.int64)

    # ── Training PyTorch ──────────────────────────────────────────────
    metrics = _train_local(
        train_features = train_features,
        train_labels   = train_labels,
        test_features  = test_features,
        test_labels    = test_labels,
        window_size    = BEST_WINDOW_SIZE,
        hidden_size    = BEST_HIDDEN_SIZE,
        num_layers     = BEST_NUM_LAYERS,
        lr             = BEST_LR,
        num_epochs     = BEST_EPOCHS,
        batch_size     = BEST_BATCH_SIZE,
        seed           = seed,
    )

    return {
        "noise_type": noise_type,
        "feature":    feature_label,
        "percentage": percentage,
        "seed":       seed,
        "f1":         metrics["f1"],
        "auc":        metrics["auc"],
        "n_train":    n_train,
        "duration_s": time.time() - t0,
    }


# =============================================================================
# LOOP SPERIMENTALE
# =============================================================================

def loop_sperimentale(
    pt_train,
    test_features: np.ndarray,
    test_labels:   np.ndarray,
    n_train_base:  int,
):
    print("\n" + "=" * 60)
    print("LOOP SPERIMENTALE — parametri attivi")
    print("=" * 60)
    print(f"  window_size : {BEST_WINDOW_SIZE}")
    print(f"  hidden_size : {BEST_HIDDEN_SIZE}")
    print(f"  num_layers  : {BEST_NUM_LAYERS}")
    print(f"  lr          : {BEST_LR}")
    print(f"  num_epochs  : {BEST_EPOCHS}")
    print(f"  batch_size  : {BEST_BATCH_SIZE}")
    print()

    # ── Resume ────────────────────────────────────────────────────────
    results_dir = os.path.dirname(os.path.abspath(RESULTS_PATH))
    os.makedirs(results_dir, exist_ok=True)

    done: set = set()
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    done.add((r["noise_type"], r["feature"], r["percentage"], r["seed"]))
                except Exception:
                    pass
        print(f"Run già completati : {len(done)}")
    else:
        print("Nessun risultato precedente — partenza da zero")

    # ── Lista run ──────────────────────────────────────────────────────
    runs_to_do = []
    for noise_type in NOISE_TYPES:
        if noise_type in NOISE_NO_FEATURE:
            for percentage in NOISE_LEVELS:
                for seed in _SEEDS:
                    if (noise_type, "all_features", percentage, seed) not in done:
                        runs_to_do.append((noise_type, "all_features", NOISE_FEATURES, percentage, seed))
        else:
            for feature in NOISE_FEATURES:
                for percentage in NOISE_LEVELS:
                    for seed in _SEEDS:
                        if (noise_type, feature, percentage, seed) not in done:
                            runs_to_do.append((noise_type, feature, [feature], percentage, seed))

    n_seeds    = len(list(_SEEDS))
    total_runs = (
        len(NOISE_NO_FEATURE) * len(NOISE_LEVELS) * n_seeds
        + (len(NOISE_TYPES) - len(NOISE_NO_FEATURE))
          * len(NOISE_FEATURES) * len(NOISE_LEVELS) * n_seeds
    )
    print(f"Run totali    : {total_runs}")
    print(f"Run rimanenti : {len(runs_to_do)}\n")

    def _sig(sig, frame):
        print("\n\n⚠️  Interruzione — uscita pulita.")
        sys.exit(0)
    signal.signal(signal.SIGINT,  _sig)
    signal.signal(signal.SIGTERM, _sig)

    t_start = time.time()

    with open(RESULTS_PATH, "a") as out_f:
        for i, (noise_type, feature_label, feature_arg, percentage, seed) in enumerate(runs_to_do):
            print(
                f"→ [{i+1}/{len(runs_to_do)}]"
                f" {noise_type:<12} | {feature_label:<30}"
                f" | {percentage:.0%} | seed={seed}",
                flush=True,
            )
            try:
                result = run_experiment(
                    pt_train      = pt_train,
                    test_features = test_features,
                    test_labels   = test_labels,
                    n_train_base  = n_train_base,
                    noise_type    = noise_type,
                    feature_arg   = feature_arg,
                    feature_label = feature_label,
                    percentage    = percentage,
                    seed          = seed,
                )
                result["timestamp"] = datetime.now().isoformat()
                out_f.write(json.dumps(result) + "\n")
                out_f.flush()

                avg_s       = (time.time() - t_start) / (i + 1)
                remaining_h = (len(runs_to_do) - i - 1) * avg_s / 3600

                print(
                    f"  ✓ F1={result['f1']:.4f}"
                    f" | AUC={result['auc']:.4f}"
                    f" | {result['duration_s']:.0f}s"
                    f" | n_train={result['n_train']:,}"
                    f" | remaining≈{remaining_h:.1f}h",
                    flush=True,
                )

            except KeyboardInterrupt:
                print("\n\n⚠️  Ctrl+C — uscita pulita.")
                sys.exit(0)
            except Exception:
                print("  ❌ ERRORE:")
                traceback.print_exc()
                print("  — continuo —\n", flush=True)
                continue

    print(f"\n✅ Loop completato — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"✅ Risultati in  : {os.path.abspath(RESULTS_PATH)}")

    print("\n--- Riepilogo aggregato ---")
    rows = []
    with open(RESULTS_PATH) as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    if rows:
        df_res  = pd.DataFrame(rows)
        summary = (
            df_res
            .groupby(["noise_type", "feature", "percentage"])[["f1", "auc"]]
            .agg(["mean", "std", "count"])
            .round(4)
        )
        print(summary.to_string())


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    spark, pt_train, test_features, test_labels, n_train_base = init_data()
    loop_sperimentale(pt_train, test_features, test_labels, n_train_base)
    spark.stop()
