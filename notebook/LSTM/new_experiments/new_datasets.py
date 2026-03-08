"""
generate_noisy_datasets_v2.py
==============================
FASE 1 — Genera i parquet aggiuntivi sul cluster Spark.

Nuovi gruppi rispetto alla v1:
  1. duplicated  — affected_features=["target"], selection_criteria="target = 1"
                   (duplica solo le righe di guasto, come nel MLP)
  2. missing     — affected_features=["TP3_scaled", "Reservoirs_scaled"]
  3. noise       — affected_features=["TP3_scaled", "Reservoirs_scaled"]
  4. outliers    — affected_features=["TP3_scaled", "Reservoirs_scaled"]

I parquet vengono salvati in NOISY_DIR con naming:
  duplicated_target1_{pct}pct/
  {noise_type}_TP3_Reservoirs_{pct}pct/

Esecuzione (sul cluster o dal driver):
    python generate_noisy_datasets_v2.py
"""

# =============================================================================
# PATH — MODIFICA QUI
# =============================================================================
DATA_PATH  = "/home/PuckTrickadmin/DATASETS/MetroDT_Modified.parquet"
NOISY_DIR  = "/home/PuckTrickadmin/DATASETS/noisy"
SPLIT_DATE = "2020-06-01 00:00:00"
LOG_PATH   = "/home/PuckTrickadmin/DATASETS/generation_log_v2.jsonl"

MASTER_URL  = "spark://10.0.1.8:7077"
DRIVER_HOST = "10.0.1.8"

# =============================================================================
import os, sys, json, time
from datetime import datetime

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

from pucktrick import PuckTrick, Engine

# =============================================================================
# Spark
# =============================================================================
spark = (
    SparkSession.builder
    .appName("MetroPT_GenerateNoisy_v2")
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
# Costanti
# =============================================================================
ALL_FEATURES = [
    "TP2_scaled", "TP3_scaled", "H1_scaled", "DV_pressure_scaled",
    "Reservoirs_scaled", "Oil_temperature_scaled", "Motor_current_scaled",
    "COMP", "DV_eletric", "Towers", "MPG", "LPS", "Pressure_switch",
]
LABEL_COL        = "target"
FEATURE_COMBINED = ["TP3_scaled", "Reservoirs_scaled"]
COMBINED_LABEL   = "TP3_Reservoirs"   # usato nel nome del parquet

NOISE_LEVELS = [0.1, 0.2, 0.3, 0.5]

# =============================================================================
# Dataset e split
# =============================================================================
print("Caricamento dataset...")
raw_pdf = pd.read_parquet(DATA_PATH)
raw_df  = spark.createDataFrame(raw_pdf)
del raw_pdf

cols_to_cast = ALL_FEATURES + [LABEL_COL]
df = raw_df.select(
    F.col("timestamp"),
    *[F.col(c).cast(DoubleType()) for c in cols_to_cast],
).dropna()

pt_base    = PuckTrick(df, engine=Engine.SPARK)
base_df    = pt_base.original
train_df   = base_df.filter(F.col("timestamp") < SPLIT_DATE).drop("timestamp")
train_df.cache()

pt_train   = PuckTrick(train_df, engine=Engine.SPARK)

n_train = pt_train.original.count()
n_fault = pt_train.original.filter(F.col(LABEL_COL) == 1).count()
print(f"Training : {n_train:,} righe  (fault={n_fault:,})")

# =============================================================================
# Resume — legge i parquet già generati
# =============================================================================
completed = set()
if os.path.exists(LOG_PATH):
    with open(LOG_PATH) as f:
        for line in f:
            try:
                completed.add(json.loads(line)["parquet_name"])
            except Exception:
                pass
    print(f"Parquet già generati: {len(completed)}")

# =============================================================================
# Piano di generazione
# =============================================================================
tasks = []

# Gruppo 1 — Duplicated con selection target=1
for pct in NOISE_LEVELS:
    pct_str = str(int(round(pct * 100)))
    name    = f"duplicated_target1_{pct_str}pct"
    tasks.append({
        "name":       name,
        "noise_type": "duplicated",
        "strategy": {
            "affected_features":  [LABEL_COL],
            "selection_criteria": f"target = 1",
            "percentage":         pct,
            "mode":               "new",
            "perturbate_data": {
                "distribution": "random",
                "param":        {"seed": 42},
            },
        },
    })

# Gruppi 2-4 — Missing / Noise / Outliers su TP3+Reservoirs
for noise_type in ["missing", "noise", "outliers"]:
    for pct in NOISE_LEVELS:
        pct_str = str(int(round(pct * 100)))
        name    = f"{noise_type}_{COMBINED_LABEL}_{pct_str}pct"
        tasks.append({
            "name":       name,
            "noise_type": noise_type,
            "strategy": {
                "affected_features":  FEATURE_COMBINED,
                "selection_criteria": "all",
                "percentage":         pct,
                "mode":               "new",
                "perturbate_data": {
                    "distribution": "random",
                    "param":        {"seed": 42},
                },
            },
        })

todo = [t for t in tasks if t["name"] not in completed]
print(f"\nTask totali : {len(tasks)}")
print(f"Da generare : {len(todo)}\n")

# =============================================================================
# Generazione
# =============================================================================
os.makedirs(NOISY_DIR, exist_ok=True)

with open(LOG_PATH, "a") as log_file:
    for i, task in enumerate(todo):
        name       = task["name"]
        noise_type = task["noise_type"]
        strategy   = task["strategy"]
        out_path   = os.path.join(NOISY_DIR, name)

        print(f"[{i+1}/{len(todo)}] Generazione: {name}")
        t0 = time.time()

        try:
            if noise_type == "duplicated":
                status, noisy_df = pt_train.duplicated(pt_train.original, strategy)
            elif noise_type == "missing":
                status, noisy_df = pt_train.missing(pt_train.original, strategy)
            elif noise_type == "noise":
                status, noisy_df = pt_train.noise(pt_train.original, strategy)
            elif noise_type == "outliers":
                status, noisy_df = pt_train.outlier(pt_train.original, strategy)
            else:
                raise ValueError(f"Tipo sconosciuto: {noise_type}")

            if status != 0:
                print(f"  ⚠️  status={status} — salto")
                continue

            # Rimuove colonne interne pucktrick prima di salvare
            cols_to_drop = [c for c in noisy_df.columns
                            if c.startswith("_pucktrick")]
            if cols_to_drop:
                noisy_df = noisy_df.drop(*cols_to_drop)

            noisy_df.write.mode("overwrite").parquet(out_path)
            elapsed = time.time() - t0
            n_rows  = noisy_df.count()
            print(f"  ✅ {n_rows:,} righe → {out_path}  ({elapsed:.1f}s)")

            log_entry = {
                "parquet_name": name,
                "out_path":     out_path,
                "n_rows":       n_rows,
                "elapsed_s":    round(elapsed, 1),
                "timestamp":    datetime.now().isoformat(),
            }
            log_file.write(json.dumps(log_entry) + "\n")
            log_file.flush()

        except Exception as e:
            import traceback
            print(f"  ❌ ERRORE: {e}")
            traceback.print_exc()
            continue

print("\n✅ Generazione completata.")
spark.stop()