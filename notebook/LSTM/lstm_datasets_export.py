"""
generate_noisy_datasets.py
===========================
FASE 1 — Genera tutti i dataset rumorosi con PuckTrick+Spark e li salva come parquet.
Da eseguire UNA SOLA VOLTA sul cluster.

Esecuzione:
    spark-submit --driver-memory 8g --executor-memory 13g generate_noisy_datasets.py

Output:
    OUTPUT_DIR/
        baseline/           ← training set pulito (0% rumore)
        test_set/           ← test set (condiviso, mai rumoroso)
        duplicated_all_10pct/
        duplicated_all_20pct/
        ...
        missing_DV_pressure_scaled_50pct/
        ...
        generation_log.jsonl
"""

# =============================================================================
# 🗺️ PATH — MODIFICA QUI
# =============================================================================
DATA_PATH  = r"/home/PuckTrickadmin/DATASETS/MetroDT_Modified.parquet"
OUTPUT_DIR = r"/home/PuckTrickadmin/DATASETS/noisy"
MASTER_URL = "spark://10.0.1.8:7077"
DRIVER_HOST = "10.0.1.8"
SPLIT_DATE  = "2020-06-01 00:00:00"

# =============================================================================
import os, json, time, sys
from datetime import datetime
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

from pucktrick import PuckTrick, Engine

# =============================================================================
# SCHEMA
# =============================================================================
LABEL_COL    = "target"
ALL_FEATURES = [
    "TP2_scaled", "TP3_scaled", "H1_scaled", "DV_pressure_scaled",
    "Reservoirs_scaled", "Oil_temperature_scaled", "Motor_current_scaled",
    "COMP", "DV_eletric", "Towers", "MPG", "LPS", "Pressure_switch",
]
NOISE_FEATURES = ["DV_pressure_scaled", "Oil_temperature_scaled", "TP3_scaled"]
NOISE_LEVELS   = [0.1, 0.2, 0.3, 0.5]

# =============================================================================
# ESPERIMENTI DA GENERARE
# duplicated → 4 parquet (feature = "all", no affected_features specifiche)
# labels     → 4 parquet (feature = "all", affected = LABEL_COL)
# missing    → 12 parquet (3 feature × 4 livelli)
# noise      → 12 parquet
# outliers   → 12 parquet
# TOTALE     → 44 parquet + baseline + test_set
# =============================================================================
EXPERIMENT_PLAN = [
    # (noise_type, feature_label, affected_features_value)
    *[("duplicated", "all",     None)          for _ in [None]],  # espanso sotto
    *[("labels",     "all",     LABEL_COL)     for _ in [None]],
    *[(nt, feat, feat)
      for nt in ["missing", "noise", "outliers"]
      for feat in NOISE_FEATURES],
]

# Espandi per i livelli di rumore
JOBS = [
    (noise_type, feature_label, affected, pct)
    for (noise_type, feature_label, affected) in EXPERIMENT_PLAN
    for pct in NOISE_LEVELS
]

# =============================================================================
# STRATEGY BUILDER
# =============================================================================
def make_strategy(noise_type: str, affected, percentage: float) -> dict:
    base = {
        "selection_criteria": "all",
        "percentage": percentage,
        "mode": "new",
        "perturbate_data": {
            "distribution": "random",
            "param": {}
        }
    }
    if noise_type == "duplicated":
        # duplicated agisce sulle righe, non richiede affected_features
        return base
    elif noise_type == "labels":
        return {**base, "affected_features": affected}
    else:
        # missing, noise, outliers
        return {
            **base,
            "affected_features": [affected],
            "perturbate_data": {
                "distribution": "random",
                "value": [None],
                "param": {}
            }
        }

# =============================================================================
# PATH HELPER
# =============================================================================
def out_path(noise_type: str, feature_label: str, percentage: float) -> str:
    pct_str = str(int(round(percentage * 100)))
    return os.path.join(OUTPUT_DIR, f"{noise_type}_{feature_label}_{pct_str}pct")

# =============================================================================
# MAIN
# =============================================================================
def main():
    if not os.path.exists(DATA_PATH):
        print(f"❌ Dataset non trovato: {DATA_PATH}")
        sys.exit(1)

    # --- SparkSession ---
    spark = SparkSession.builder \
        .appName("PuckTrick_GenerateNoisy") \
        .master(MASTER_URL) \
        .config("spark.submit.deployMode",      "client") \
        .config("spark.executor.instances",     "4") \
        .config("spark.executor.cores",         "4") \
        .config("spark.executor.memory",        "13g") \
        .config("spark.driver.memory",          "8g") \
        .config("spark.driver.host",            DRIVER_HOST) \
        .config("spark.driver.bindAddress",     DRIVER_HOST) \
        .config("spark.sql.shuffle.partitions", "32") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    print(f"✅ SparkSession {spark.version}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Carica dataset ---
    print("📥 Caricamento dataset Pandas...")
    raw_pdf = pd.read_parquet(DATA_PATH)
    raw_pdf["timestamp"] = pd.to_datetime(raw_pdf["timestamp"]).dt.strftime('%Y-%m-%d %H:%M:%S')

    schema = StructType([
        StructField("timestamp", StringType(), True),
        *[StructField(c, DoubleType(), True) for c in ALL_FEATURES],
        StructField(LABEL_COL, DoubleType(), True),
    ])

    raw_df   = spark.createDataFrame(raw_pdf, schema=schema).dropna()
    train_df = raw_df.filter(F.col("timestamp") < SPLIT_DATE)
    test_df  = raw_df.filter(F.col("timestamp") >= SPLIT_DATE)
    n_train, n_test = train_df.count(), test_df.count()
    print(f"   Train: {n_train:,} | Test: {n_test:,}")

    # --- Salva baseline e test set ---
    baseline_path = os.path.join(OUTPUT_DIR, "baseline")
    test_path     = os.path.join(OUTPUT_DIR, "test_set")

    for path, df, label in [(baseline_path, train_df, "baseline"),
                             (test_path,     test_df,  "test_set")]:
        if not os.path.exists(path):
            print(f"💾 Salvataggio {label}...")
            df.orderBy("timestamp").write.mode("overwrite").parquet(path)
            print(f"   ✅ {label} → {path}")
        else:
            print(f"   ⏭️  {label} già esistente")

    # --- Init PuckTrick ---
    pt = PuckTrick(train_df, engine=Engine.SPARK)
    base_df = pt.original
    base_df.cache()
    print(f"✅ PuckTrick inizializzato su {base_df.count():,} righe")

    METHOD_MAP = {
        "duplicated": pt.duplicated,
        "labels":     pt.labels,
        "missing":    pt.missing,
        "noise":      pt.noise,
        "outliers":   pt.outlier,
    }

    # --- Resume: carica log ---
    log_path  = os.path.join(OUTPUT_DIR, "generation_log.jsonl")
    completed = set()
    if os.path.exists(log_path):
        with open(log_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    completed.add((r["noise_type"], r["feature"], r["percentage"]))
                except Exception:
                    pass
        print(f"✅ Già completati: {len(completed)} job")

    # --- Loop generazione ---
    total = len(JOBS)
    print(f"\n{'='*60}")
    print(f"GENERAZIONE: {total} dataset rumorosi")
    print(f"{'='*60}\n")

    with open(log_path, "a") as log_file:
        for idx, (noise_type, feature_label, affected, pct) in enumerate(JOBS, 1):
            key      = (noise_type, feature_label, pct)
            dst_path = out_path(noise_type, feature_label, pct)

            if key in completed or os.path.exists(dst_path):
                print(f"  ⏭️  [{idx:02d}/{total}] {noise_type:<12} | {feature_label:<25} | {int(pct*100):>3}%  (skip)")
                continue

            t0 = time.time()
            try:
                strategy = make_strategy(noise_type, affected, pct)
                status, noisy_df = METHOD_MAP[noise_type](base_df, strategy)
                noisy_df.orderBy("timestamp").write.mode("overwrite").parquet(dst_path)
                elapsed = time.time() - t0

                log_entry = {
                    "noise_type":  noise_type,
                    "feature":     feature_label,
                    "percentage":  pct,
                    "status":      status,
                    "duration_s":  round(elapsed, 2),
                    "output_path": dst_path,
                    "timestamp":   datetime.now().isoformat(),
                }
                log_file.write(json.dumps(log_entry) + "\n")
                log_file.flush()

                print(f"  ✅ [{idx:02d}/{total}] {noise_type:<12} | {feature_label:<25} | {int(pct*100):>3}%"
                      f"  → {elapsed:.1f}s  ({dst_path})")

            except Exception as e:
                print(f"  ❌ [{idx:02d}/{total}] {noise_type:<12} | {feature_label:<25} | {int(pct*100):>3}%"
                      f"  ERRORE: {e}")

    base_df.unpersist()
    spark.stop()
    print(f"\n✅ Generazione completata → {OUTPUT_DIR}")
    print(f"   Riepilogo spazio: du -sh {OUTPUT_DIR}")

if __name__ == "__main__":
    main()