"""
lstm-esperimenti.py
===================
Loop sperimentale LSTM.

Struttura run:
- labels, duplicated : 5 livelli × 20 run = 100 run ciascuno (no feature loop)
- missing, noise, outliers : 3 feature × 5 livelli × 20 run = 300 run ciascuno

Totale: 2 × 100 + 3 × 300 = 1.100 run

Prerequisito: aver eseguito lstm-tuning.py e impostato i BEST_* qui sotto.
Esecuzione  : python lstm_model.py
"""
import os
os.environ["HADOOP_HOME"] = "C:\\hadoop"
os.environ["PATH"] = os.environ["PATH"] + ";C:\\hadoop\\bin"
os.environ["PYSPARK_PYTHON"] = "D:\\Users\\satri\\Pictures\\github\\Deep-Learning-Robustness-Study\\.venv\\Scripts\\python.exe"
os.environ["PYSPARK_DRIVER_PYTHON"] = "D:\\Users\\satri\\Pictures\\github\\Deep-Learning-Robustness-Study\\.venv\\Scripts\\python.exe"


import time
import json
import signal
import sys
import traceback
import os
from datetime import datetime

import pandas as pd
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.ml.torch.distributor import TorchDistributor

from pucktrick import PuckTrick, Engine
from sklearn.metrics import f1_score, roc_auc_score

# =============================================================================
# ██ BEST PARAMS — IMPOSTARE MANUALMENTE DAI RISULTATI DI lstm-tuning.py ██
# =============================================================================

BEST_WINDOW_SIZE: int = 30
BEST_HIDDEN_SIZE: int = 128
BEST_NUM_LAYERS: int  = 1
BEST_LR: float        = 1e-3
BEST_EPOCHS: int      = 10
BEST_BATCH_SIZE: int  = 512

# =============================================================================

ALL_FEATURES = [
    "TP2_scaled", "TP3_scaled", "H1_scaled", "DV_pressure_scaled",
    "Reservoirs_scaled", "Oil_temperature_scaled", "Motor_current_scaled",
    "COMP", "DV_eletric", "Towers", "MPG", "LPS", "Pressure_switch"
]
NOISE_FEATURES   = ["DV_pressure_scaled", "Oil_temperature_scaled", "TP3_scaled"]
NOISE_NO_FEATURE = ["labels", "duplicated"]
LABEL_COL        = "target"
N_FEATURES       = len(ALL_FEATURES)  # 13

NOISE_TYPES  = ["duplicated", "labels", "missing", "noise", "outliers"]
NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.5]
N_RUNS       = 20
T_VALUE_95   = 2.093

RESULTS_PATH = "D:/Users/satri/Pictures/github/Deep-Learning-Robustness-Study/RESULTS/lstm_results.jsonl"
TMP_DATA_DIR = "D:/Users/satri/Pictures/github/Deep-Learning-Robustness-Study/tmp"
VENV_PYTHON  = "D:/Users/satri/Pictures/github/Deep-Learning-Robustness-Study/.venv/Scripts/python.exe"
DATA_PATH    = "D:\\Users\\satri\\Pictures\\github\\Deep-Learning-Robustness-Study\\notebook\\Dataset\\MetroDT_Modified.parquet"

# =============================================================================
# FUNZIONE DI TRAINING DISTRIBUITO
# =============================================================================

def _distributed_train_fn(
    data_path:   str,
    window_size: int,
    hidden_size: int,
    num_layers:  int,
    lr:          float,
    num_epochs:  int,
    batch_size:  int,
    n_features:  int,
    seed:        int,
):
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import Dataset, DataLoader
    from torch.utils.data.distributed import DistributedSampler
    import numpy as np
    from sklearn.metrics import f1_score, roc_auc_score

    dist.init_process_group(backend="gloo")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)

    data           = np.load(data_path)
    train_features = data["train_features"]
    train_labels   = data["train_labels"]
    test_features  = data["test_features"]
    test_labels    = data["test_labels"]

    class _WinDS(Dataset):
        def __init__(self, feats, labs, ws):
            self.f  = torch.tensor(feats, dtype=torch.float32)
            self.l  = torch.tensor(labs,  dtype=torch.long)
            self.ws = ws
        def __len__(self):
            return len(self.f) - self.ws + 1
        def __getitem__(self, i):
            return self.f[i : i + self.ws], self.l[i + self.ws - 1]

    train_ds     = _WinDS(train_features, train_labels, window_size)
    sampler      = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=seed
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)

    class _LSTM(nn.Module):
        def __init__(self):
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

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = DDP(_LSTM().to(device))
    n_pos     = max((train_labels == 1).sum(), 1)
    n_neg     = (train_labels == 0).sum()
    weight    = torch.tensor([1.0, n_neg / n_pos], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

    metrics = {"f1": 0.0, "auc": 0.0}
    if rank == 0:
        test_ds     = _WinDS(test_features, test_labels, window_size)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        model.eval()
        preds, probs, labs = [], [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                logits = model(xb)
                probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
                preds.extend(logits.argmax(dim=1).cpu().numpy())
                labs.extend(yb.numpy())

        labs  = np.array(labs)
        preds = np.array(preds)
        probs = np.array(probs)

        f1 = f1_score(labs, preds, average="weighted", zero_division=0)
        try:
            auc = roc_auc_score(labs, probs)
        except ValueError:
            auc = 0.0

        metrics = {"f1": f1, "auc": auc}

    dist.destroy_process_group()
    return metrics

# =============================================================================
# INIT SPARK
# =============================================================================

def init_spark():
    spark = SparkSession.builder \
        .appName("MetroPT_LSTM_Experiments") \
        .master("local[*]") \
        .config("spark.driver.memory",          "8g") \
        .config("spark.driver.host",            "127.0.0.1") \
        .config("spark.driver.bindAddress",     "127.0.0.1") \
        .config("spark.sql.shuffle.partitions", "32") \
        .config("spark.pyspark.python",         VENV_PYTHON) \
        .config("spark.pyspark.driver.python",  VENV_PYTHON) \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    print("SparkSession creata — versione:", spark.version)

    raw_pdf = pd.read_parquet(DATA_PATH)
    raw_df  = spark.createDataFrame(raw_pdf)
    del raw_pdf

    df = raw_df.select(
        F.col("timestamp"),
        *[F.col(c).cast(DoubleType()) for c in ALL_FEATURES],
        F.col("target").cast(DoubleType()),
    ).dropna()

    SPLIT_DATE    = "2020-06-01 00:00:00"
    pt            = PuckTrick(df, engine=Engine.SPARK)
    base_df       = pt.original

    base_train_df = base_df.filter(F.col("timestamp") <  SPLIT_DATE)
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

    train_df_with_ts = base_df.filter(F.col("timestamp") < SPLIT_DATE)
    pt_train         = PuckTrick(train_df_with_ts, engine=Engine.SPARK)

    return spark, base_train_df, base_test_df, pt_train

# =============================================================================
# SINGOLO RUN SPERIMENTALE
# =============================================================================

def run_experiment(
    pt_train,
    base_test_df,
    noise_type:  str,
    feature:     str,
    percentage:  float,
    seed:        int,
    window_size: int   = BEST_WINDOW_SIZE,
    hidden_size: int   = BEST_HIDDEN_SIZE,
    num_layers:  int   = BEST_NUM_LAYERS,
    lr:          float = BEST_LR,
    num_epochs:  int   = BEST_EPOCHS,
    batch_size:  int   = BEST_BATCH_SIZE,
) -> dict:
    t0 = time.time()
    os.makedirs(TMP_DATA_DIR, exist_ok=True)

    if percentage == 0.0:
        noisy_df = pt_train.original
    else:
        strategy = {
            "affected_features":  [feature],
            "selection_criteria": "all",
            "percentage":         percentage,
            "mode":               "new",
            "perturbate_data": {
                "distribution": "random",
                "param":        {"seed": seed}
            }
        }

        if noise_type == "duplicated":
            status, noisy_df = pt_train.duplicated(pt_train.original, strategy)
        elif noise_type == "labels":
            label_strategy   = {**strategy, "affected_features": [LABEL_COL]}
            status, noisy_df = pt_train.labels(pt_train.original, label_strategy)
        elif noise_type == "missing":
            status, noisy_df = pt_train.missing(pt_train.original, strategy)
        elif noise_type == "noise":
            status, noisy_df = pt_train.noise(pt_train.original, strategy)
        elif noise_type == "outliers":
            status, noisy_df = pt_train.outlier(pt_train.original, strategy)
        else:
            raise ValueError(f"Tipo di rumore non supportato: {noise_type}")

        if status != 0:
            print(f"⚠️  inject_noise: status={status} ({noise_type}, {feature}, {percentage:.0%}, seed={seed})")
            noisy_df = pt_train.original

    noisy_df = noisy_df.orderBy("timestamp").drop("timestamp")
    noisy_df.cache()
    n_train  = noisy_df.count() if noise_type == "duplicated" else pt_train.original.count()

    train_pdf = noisy_df.select(ALL_FEATURES + [LABEL_COL]).toPandas()
    test_pdf  = (
        base_test_df
        .orderBy("timestamp")
        .select(ALL_FEATURES + [LABEL_COL])
        .toPandas()
    )

    train_features = train_pdf[ALL_FEATURES].values.astype(np.float32)
    train_labels   = train_pdf[LABEL_COL].values.astype(np.int64)
    test_features  = test_pdf[ALL_FEATURES].values.astype(np.float32)
    test_labels    = test_pdf[LABEL_COL].values.astype(np.int64)

    data_path = os.path.join(TMP_DATA_DIR, f"run_{noise_type}_{feature}_{seed}.npz")
    np.savez(
        data_path,
        train_features = train_features,
        train_labels   = train_labels,
        test_features  = test_features,
        test_labels    = test_labels,
    )

    distributor = TorchDistributor(
        num_processes = 1,
        local_mode    = True,
        use_gpu       = True,
    )
    metrics = distributor.run(
        _distributed_train_fn,
        data_path,
        window_size,
        hidden_size,
        num_layers,
        lr,
        num_epochs,
        batch_size,
        N_FEATURES,
        seed,
    )

    try:
        os.remove(data_path)
    except OSError:
        pass
    noisy_df.unpersist()

    return {
        "noise_type": noise_type,
        "feature":    feature,
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
    base_test_df,
    window_size: int   = BEST_WINDOW_SIZE,
    hidden_size: int   = BEST_HIDDEN_SIZE,
    num_layers:  int   = BEST_NUM_LAYERS,
    lr:          float = BEST_LR,
    num_epochs:  int   = BEST_EPOCHS,
    batch_size:  int   = BEST_BATCH_SIZE,
):
    print("\n" + "="*60)
    print("LOOP SPERIMENTALE — parametri attivi")
    print("="*60)
    print(f"  window_size : {window_size}")
    print(f"  hidden_size : {hidden_size}")
    print(f"  num_layers  : {num_layers}")
    print(f"  lr          : {lr}")
    print(f"  num_epochs  : {num_epochs}")
    print(f"  batch_size  : {batch_size}")
    print()

    all_runs = []
    for noise_type in NOISE_TYPES:
        if noise_type in NOISE_NO_FEATURE:
            for percentage in NOISE_LEVELS:
                for seed in range(N_RUNS):
                    all_runs.append((noise_type, "all_features", percentage, seed))
        else:
            for feature in NOISE_FEATURES:
                for percentage in NOISE_LEVELS:
                    for seed in range(N_RUNS):
                        all_runs.append((noise_type, feature, percentage, seed))

    total_runs = (len(NOISE_NO_FEATURE) * len(NOISE_LEVELS) * N_RUNS) + \
                 ((len(NOISE_TYPES) - len(NOISE_NO_FEATURE)) * len(NOISE_FEATURES) * len(NOISE_LEVELS) * N_RUNS)

    done = set()
    try:
        with open(RESULTS_PATH) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    done.add((r["noise_type"], r["feature"], r["percentage"], r["seed"]))
                except Exception:
                    pass
        print(f"Run già completati: {len(done)}")
    except FileNotFoundError:
        pass

    runs_to_do = [r for r in all_runs if (r[0], r[1], r[2], r[3]) not in done]
    print(f"Run totali      : {total_runs}")
    print(f"Run da eseguire : {len(runs_to_do)} / {total_runs}\n")

    def _sig(sig, frame):
        print("\n\n⚠️  Interruzione — uscita pulita.")
        sys.exit(0)
    signal.signal(signal.SIGINT,  _sig)
    signal.signal(signal.SIGTERM, _sig)

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

    with open(RESULTS_PATH, "a") as out_f:
        for i, (noise_type, feature, percentage, seed) in enumerate(runs_to_do):
            try:
                feature_arg = NOISE_FEATURES[0] if noise_type in NOISE_NO_FEATURE else feature

                result = run_experiment(
                    pt_train     = pt_train,
                    base_test_df = base_test_df,
                    noise_type   = noise_type,
                    feature      = feature_arg,
                    percentage   = percentage,
                    seed         = seed,
                    window_size  = window_size,
                    hidden_size  = hidden_size,
                    num_layers   = num_layers,
                    lr           = lr,
                    num_epochs   = num_epochs,
                    batch_size   = batch_size,
                )
                result["feature"]   = feature
                result["timestamp"] = datetime.now().isoformat()
                out_f.write(json.dumps(result) + "\n")
                out_f.flush()

                if (i + 1) % 10 == 0:
                    avg_dur  = result["duration_s"]
                    remain_h = (len(runs_to_do) - i - 1) * avg_dur / 3600
                    print(
                        f"[{datetime.now().strftime('%H:%M:%S')}]"
                        f"  Run {i+1}/{len(runs_to_do)}"
                        f"  {noise_type:<12}"
                        f"  {feature:<25}"
                        f"  {percentage:.0%}"
                        f"  seed={seed}"
                        f"  F1={result['f1']:.4f}"
                        f"  AUC={result['auc']:.4f}"
                        f"  n_train={result['n_train']:,}"
                        f"  remaining≈{remain_h:.1f}h"
                    )

            except KeyboardInterrupt:
                print("\n\n⚠️  Ctrl+C — uscita pulita.")
                sys.exit(0)

            except Exception:
                print(f"\n❌ ERRORE ({noise_type}, {feature}, {percentage:.0%}, seed={seed}):")
                traceback.print_exc()
                print("— continuo con il prossimo run —\n")
                continue

    print(f"\n✅ Loop completato — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"✅ Risultati in: {RESULTS_PATH}")

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
            df_res.groupby(["noise_type", "feature", "percentage"])[["f1", "auc"]]
            .agg(["mean", "std", "count"])
            .round(4)
        )
        print(summary.to_string())

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    spark, base_train_df, base_test_df, pt_train = init_spark()
    loop_sperimentale(pt_train, base_test_df)
    spark.stop()
