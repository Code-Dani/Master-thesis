"""
mlp_new_experiments.py
======================
Esperimenti MLP aggiuntivi su MetroPT-3 tramite PuckTrick + Apache Spark.

Gruppi:
  1. Duplicated  — affected_features=["target"], selection_criteria="target = 1"
  2. Missing     — affected_features=FEATURE_COMBINED ["TP3_scaled","Reservoirs_scaled"]
  3. Noisy       — affected_features=FEATURE_COMBINED
  4. Outliers    — affected_features=FEATURE_COMBINED

La baseline (0%) NON viene rieseguita (già in mlp_results.jsonl).
Resume automatico su mlp_results_new.jsonl.

Esecuzione:
    python mlp_new_experiments.py
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from pyspark.ml import Pipeline
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.ml.feature import Imputer, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType


from pucktrick import PuckTrick, Engine

# =============================================================================
# STEP 1 — Spark
# =============================================================================

MASTER_URL  = "spark://10.0.1.8:7077"
DRIVER_HOST = "10.0.1.8"

spark = (
    SparkSession.builder
    .appName("MetroPT_MLP_NewExperiments")
    .master(MASTER_URL)
    .config("spark.submit.deployMode",      "client")
    .config("spark.executor.instances",     "4")
    .config("spark.executor.cores",         "4")
    .config("spark.executor.memory",        "13g")
    .config("spark.driver.memory",          "8g")
    .config("spark.driver.host",            DRIVER_HOST)
    .config("spark.driver.bindAddress",     DRIVER_HOST)
    .config("spark.sql.shuffle.partitions", "32")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")
print(f"SparkSession avviata — versione: {spark.version}")

# =============================================================================
# STEP 2 — Costanti
# =============================================================================

ALL_FEATURES = [
    "TP2_scaled", "TP3_scaled", "H1_scaled", "DV_pressure_scaled",
    "Reservoirs_scaled", "Oil_temperature_scaled", "Motor_current_scaled",
    "COMP", "DV_eletric", "Towers", "MPG", "LPS",
]

LABEL_COL        = "target"
FEATURE_COMBINED = ["TP3_scaled", "Reservoirs_scaled"]
COMBINED_LABEL   = "+".join(FEATURE_COMBINED)   # "TP3_scaled+Reservoirs_scaled"

NOISE_LEVELS = [0.1, 0.2, 0.3, 0.5]
N_RUNS       = 20

# Iperparametri MLP — NON modificare
BEST_LAYERS   = [12, 64, 2]
BEST_MAX_ITER = 100
BEST_STEP     = 0.05
BEST_BLOCK    = 128

RESULTS_PATH_NEW = "/home/PuckTrickadmin/Daniel/RESULTS/mlp_results_new.jsonl"
os.makedirs(os.path.dirname(RESULTS_PATH_NEW), exist_ok=True)

# =============================================================================
# STEP 3 — Dataset e split temporale
# =============================================================================

DATA_PATH  = "/home/PuckTrickadmin/DATASETS/MetroDT_Modified.parquet"
SPLIT_DATE = "2020-06-01 00:00:00"

print("Caricamento dataset...")
raw_pdf = pd.read_parquet(DATA_PATH)
raw_df  = spark.createDataFrame(raw_pdf)
del raw_pdf

df = raw_df.select(
    F.col("timestamp"),
    F.col("TP2_scaled").cast(DoubleType()),
    F.col("TP3_scaled").cast(DoubleType()),
    F.col("H1_scaled").cast(DoubleType()),
    F.col("DV_pressure_scaled").cast(DoubleType()),
    F.col("Reservoirs_scaled").cast(DoubleType()),
    F.col("Oil_temperature_scaled").cast(DoubleType()),
    F.col("Motor_current_scaled").cast(DoubleType()),
    F.col("COMP").cast(DoubleType()),
    F.col("DV_eletric").cast(DoubleType()),
    F.col("Towers").cast(DoubleType()),
    F.col("MPG").cast(DoubleType()),
    F.col("LPS").cast(DoubleType()),
    F.col("target").cast(DoubleType()),
).dropna()

pt_base = PuckTrick(df, engine=Engine.SPARK)
base_df = pt_base.original

base_train_df = base_df.filter(F.col("timestamp") < SPLIT_DATE).drop("timestamp")
base_test_df  = base_df.filter(F.col("timestamp") >= SPLIT_DATE).drop("timestamp")

base_train_df.cache()
base_test_df.cache()

n_train = base_train_df.count()
n_test  = base_test_df.count()
print(f"Training : {n_train:,} righe")
print(f"Test     : {n_test:,} righe")

pt_train     = PuckTrick(base_train_df, engine=Engine.SPARK)
N_TRAIN_BASE = pt_train.original.count()
print(f"N_TRAIN_BASE: {N_TRAIN_BASE:,}")

# =============================================================================
# STEP 4 — Pipeline ML
# =============================================================================

base_imputer = Imputer(
    inputCols  = ALL_FEATURES,
    outputCols = ALL_FEATURES,
    strategy   = "mean",
)

base_assembler = VectorAssembler(
    inputCols    = ALL_FEATURES,
    outputCol    = "features",
    handleInvalid= "keep",
)

base_mlp = MultilayerPerceptronClassifier(
    featuresCol = "features",
    labelCol    = LABEL_COL,
    layers      = BEST_LAYERS,
    maxIter     = BEST_MAX_ITER,
    stepSize    = BEST_STEP,
    blockSize   = BEST_BLOCK,
)

f1_evaluator = MulticlassClassificationEvaluator(
    labelCol      = LABEL_COL,
    predictionCol = "prediction",
    metricName    = "f1",
)

auc_evaluator = BinaryClassificationEvaluator(
    labelCol         = LABEL_COL,
    rawPredictionCol = "rawPrediction",
    metricName       = "areaUnderROC",
)

# =============================================================================
# STEP 5 — inject_noise e run_experiment
# =============================================================================

def inject_noise(pt, train_df, noise_type, feature, percentage, seed):
    """
    Args:
        feature : str o list
                  - duplicated  → LABEL_COL ("target"), selection_criteria="target = 1"
                  - gli altri   → FEATURE_COMBINED (list)
    """
    if percentage == 0.0:
        return train_df

    affected = feature if isinstance(feature, list) else [feature]

    strategy = {
        "affected_features": affected,
        "selection_criteria": "all",
        "percentage":         percentage,
        "mode":               "new",
        "perturbate_data": {
            "distribution": "random",
            "param":        {"seed": seed},
        },
    }

    if noise_type == "duplicated":
        # Duplica solo le righe della classe minoritaria (fault = 1).
        # affected_features=["target"] + selection_criteria="target = 1"
        dup_strategy = {
            **strategy,
            "selection_criteria": "target = 1",
        }
        status, noisy_df = pt.duplicated(train_df, dup_strategy)

    elif noise_type == "missing":
        status, noisy_df = pt.missing(train_df, strategy)

    elif noise_type == "noisy":
        status, noisy_df = pt.noise(train_df, strategy)

    elif noise_type == "outliers":
        status, noisy_df = pt.outlier(train_df, strategy)

    else:
        raise ValueError(f"Tipo di rumore non supportato: {noise_type}")

    if status != 0:
        print(f"⚠️  inject_noise: status={status} "
              f"({noise_type}, {feature}, {percentage:.0%}, seed={seed})")
        return train_df

    return noisy_df


def run_experiment(pt, noise_type, feature, feature_label, percentage, seed):
    t0 = time.time()

    noisy_train = inject_noise(
        pt         = pt,
        train_df   = pt.original,
        noise_type = noise_type,
        feature    = feature,
        percentage = percentage,
        seed       = seed,
    )

    mlp_run  = base_mlp.setSeed(seed * 100)
    pipeline = Pipeline(stages=[base_imputer, base_assembler, mlp_run])
    model    = pipeline.fit(noisy_train)

    predictions = model.transform(base_test_df)
    f1  = f1_evaluator.evaluate(predictions)
    auc = auc_evaluator.evaluate(predictions)

    duration = time.time() - t0
    n_train  = noisy_train.count() if noise_type == "duplicated" else N_TRAIN_BASE

    return {
        "noise_type": noise_type,
        "feature":    feature_label,
        "percentage": percentage,
        "seed":       seed,
        "f1":         f1,
        "auc":        auc,
        "n_train":    n_train,
        "duration_s": duration,
    }

# =============================================================================
# STEP 6 — Lista run
# Struttura: (noise_type, feature_label, feature_arg, percentage, seed)
# =============================================================================

all_runs_new = []

# GRUPPO 1 — Duplicated: affected_features=["target"], selection="target = 1"
for pct in NOISE_LEVELS:
    for seed in range(1, N_RUNS + 1):
        all_runs_new.append(("duplicated", LABEL_COL, LABEL_COL, pct, seed))

# GRUPPO 2 — Missing / Noisy / Outliers su FEATURE_COMBINED
for noise_type in ["missing", "noisy", "outliers"]:
    for pct in NOISE_LEVELS:
        for seed in range(1, N_RUNS + 1):
            all_runs_new.append((noise_type, COMBINED_LABEL, FEATURE_COMBINED, pct, seed))

total_runs_new = len(all_runs_new)

# =============================================================================
# STEP 7 — Resume
# =============================================================================

completed_new = set()
if os.path.exists(RESULTS_PATH_NEW):
    with open(RESULTS_PATH_NEW, "r") as f:
        for line in f:
            try:
                r = json.loads(line)
                completed_new.add((r["noise_type"], r["feature"], r["percentage"], r["seed"]))
            except Exception:
                pass
    print(f"✅ Run già completati  : {len(completed_new)}")
else:
    print("✅ Nessun risultato precedente — partenza da zero")

runs_to_do = [
    (nt, fl, fa, pct, seed)
    for (nt, fl, fa, pct, seed) in all_runs_new
    if (nt, fl, pct, seed) not in completed_new
]

remaining = len(runs_to_do)
print(f"✅ Run totali previsti : {total_runs_new}")
print(f"✅ Run rimanenti       : {remaining}")
print(f"✅ Stima tempo         : {remaining * 60 / 3600:.1f} ore")
print(f"✅ Avvio               : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# =============================================================================
# STEP 8 — Loop
# =============================================================================

import signal
import traceback

def _signal_handler(sig, frame):
    print("\n\n⚠️  Interruzione manuale (Ctrl+C) — uscita pulita.")
    spark.stop()
    sys.exit(0)

signal.signal(signal.SIGINT,  _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

with open(RESULTS_PATH_NEW, "a") as results_file:
    for i, (noise_type, feature_label, feature_arg, percentage, seed) in enumerate(runs_to_do):
        try:
            result = run_experiment(
                pt            = pt_train,
                noise_type    = noise_type,
                feature       = feature_arg,
                feature_label = feature_label,
                percentage    = percentage,
                seed          = seed,
            )
            result["timestamp"] = datetime.now().isoformat()

            results_file.write(json.dumps(result) + "\n")
            results_file.flush()

            if (i + 1) % 10 == 0:
                avg_dur   = result["duration_s"]
                elapsed_h = (i + 1) * avg_dur / 3600
                remain_h  = (remaining - i - 1) * avg_dur / 3600
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"Run {i+1}/{remaining} | "
                    f"{noise_type:<12} | {feature_label:<35} | "
                    f"{percentage:.0%} | seed={seed} | "
                    f"F1={result['f1']:.4f} | AUC={result['auc']:.4f} | "
                    f"n_train={result['n_train']:,} | "
                    f"elapsed≈{elapsed_h:.1f}h | remaining≈{remain_h:.1f}h"
                )

        except KeyboardInterrupt:
            # Ctrl+C — termina immediatamente senza swallowing
            print("\n\n⚠️  Interruzione manuale (Ctrl+C) — uscita pulita.")
            spark.stop()
            sys.exit(0)

        except Exception:
            # Errore generico — stampa il traceback completo e continua
            print(f"\n❌ ERRORE ({noise_type}, {feature_label}, {percentage:.0%}, seed={seed}):")
            traceback.print_exc()
            print("— continuo con il prossimo run —\n")
            continue

print(f"\n✅ Loop completato — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"✅ Risultati in: {RESULTS_PATH_NEW}")

# =============================================================================
# STEP 9 — Aggregazione finale
# =============================================================================

print("\n--- Riepilogo aggregato ---")
rows = []
with open(RESULTS_PATH_NEW) as f:
    for line in f:
        try:
            rows.append(json.loads(line))
        except Exception:
            pass

if rows:
    df_res = pd.DataFrame(rows)
    summary = (
        df_res.groupby(["noise_type", "feature", "percentage"])[["f1", "auc"]]
        .agg(["mean", "std", "count"])
        .round(4)
    )
    print(summary.to_string())

spark.stop()