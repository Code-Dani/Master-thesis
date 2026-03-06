"""
lstm-esperimenti.py
===================
Loop sperimentale LSTM — 20 run × 5 noise type × 5 livelli = 2.000 run totali.
Il training di ogni run avviene tramite TorchDistributor (PyTorch distribuito
su cluster Spark, backend Gloo CPU).

Prerequisito
------------
Aver eseguito lstm-tuning.py e impostato i BEST_* qui sotto.

Esecuzione: python lstm-esperimenti.py
"""

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
# ██  BEST PARAMS — IMPOSTARE MANUALMENTE DAI RISULTATI DI lstm-tuning.py  ██
# =============================================================================

BEST_WINDOW_SIZE: int   = 30       # ← da sostituire
BEST_HIDDEN_SIZE: int   = 128      # ← da sostituire
BEST_NUM_LAYERS:  int   = 1        # ← da sostituire
BEST_LR:          float = 1e-3     # ← da sostituire
BEST_EPOCHS:      int   = 10       # ← da sostituire
BEST_BATCH_SIZE:  int   = 512      # fisso — cambia solo se OOM

# =============================================================================

ALL_FEATURES = [
    "TP2_scaled", "TP3_scaled", "H1_scaled", "DV_pressure_scaled",
    "Reservoirs_scaled", "Oil_temperature_scaled", "Motor_current_scaled",
    "COMP", "DV_eletric", "Towers", "MPG", "LPS", "Pressure_switch"
]
NOISE_FEATURES = ["DV_pressure_scaled", "Oil_temperature_scaled", "TP3_scaled"]
LABEL_COL      = "target"
N_FEATURES     = len(ALL_FEATURES)  # 13

NOISE_TYPES  = ["duplicated", "labels", "missing", "noisy", "outliers"]
NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.5]
N_RUNS       = 20
T_VALUE_95   = 2.093

RESULTS_PATH = "/home/PuckTrickadmin/RESULTS/lstm_results.jsonl"
TMP_DATA_DIR = "/home/PuckTrickadmin/tmp"   # filesystem condiviso — leggibile da tutti i worker


# =============================================================================
# FUNZIONE DI TRAINING DISTRIBUITO (eseguita da TorchDistributor su ogni worker)
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
    """
    Funzione chiamata da TorchDistributor su ciascun processo del cluster.

    - Carica i dati da data_path (filesystem condiviso NFS)
    - Inizializza il process group gloo (CPU)
    - Avvolge il modello in DistributedDataParallel
    - Usa DistributedSampler per partizionare i dati tra i worker
    - Solo rank 0 esegue la valutazione e restituisce le metriche
    """
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import Dataset, DataLoader
    from torch.utils.data.distributed import DistributedSampler
    import numpy as np
    from sklearn.metrics import f1_score, roc_auc_score

    # ── Init process group ────────────────────────────────────────────────
    dist.init_process_group(backend="gloo")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)

    # ── Carica dati dal filesystem condiviso ──────────────────────────────
    data           = np.load(data_path)
    train_features = data["train_features"]   # (N, F)
    train_labels   = data["train_labels"]     # (N,)
    test_features  = data["test_features"]    # (M, F)
    test_labels    = data["test_labels"]      # (M,)

    # ── Dataset con windowing temporale ───────────────────────────────────
    class _WinDS(Dataset):
        def __init__(self, feats, labs, ws):
            self.f, self.l, self.ws = (
                torch.tensor(feats, dtype=torch.float32),
                torch.tensor(labs,  dtype=torch.long),
                ws,
            )
        def __len__(self):
            return len(self.f) - self.ws + 1
        def __getitem__(self, i):
            return self.f[i : i + self.ws], self.l[i + self.ws - 1]

    train_ds = _WinDS(train_features, train_labels, window_size)

    # ── Distributed sampler — partiziona le finestre tra i worker ─────────
    sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank,
        shuffle=True, seed=seed,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler, num_workers=0,
    )

    # ── Modello + DDP ─────────────────────────────────────────────────────
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

    model = DDP(_LSTM())

    # ── Loss pesata ────────────────────────────────────────────────────────
    n_pos  = max((train_labels == 1).sum(), 1)
    n_neg  = (train_labels == 0).sum()
    weight = torch.tensor([1.0, n_neg / n_pos], dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ── Training ────────────────────────────────────────────────────────────
    model.train()
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        for xb, yb in train_loader:
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

    # ── Valutazione — solo rank 0 ─────────────────────────────────────────
    metrics = {"f1": 0.0, "auc": 0.0}
    if rank == 0:
        test_ds     = _WinDS(test_features, test_labels, window_size)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        model.eval()
        preds, probs, labs = [], [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                logits = model(xb)
                probs.extend(torch.softmax(logits, dim=1)[:, 1].numpy())
                preds.extend(logits.argmax(dim=1).numpy())
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
    return metrics   # TorchDistributor restituisce il valore di rank 0


# =============================================================================
# INIT SPARK
# =============================================================================

def init_spark():
    MASTER_URL  = "spark://10.0.1.8:7077"
    DRIVER_HOST = "10.0.1.8"

    spark = SparkSession.builder \
        .appName("MetroPT_LSTM_Experiments") \
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

    SPLIT_DATE = "2020-06-01 00:00:00"
    pt         = PuckTrick(df, engine=Engine.SPARK)
    base_df    = pt.original

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

    # PuckTrick sul training set con timestamp (necessario per iniezione rumore)
    train_df_with_ts = base_df.filter(F.col("timestamp") < SPLIT_DATE)
    pt_train = PuckTrick(train_df_with_ts, engine=Engine.SPARK)

    return spark, base_train_df, base_test_df, pt_train


# =============================================================================
# SINGOLO RUN SPERIMENTALE
# =============================================================================

def run_experiment(
    pt_train,
    base_test_df,
    noise_type:    str,
    feature:       list,
    feature_label: str,
    percentage:    float,
    seed:          int,
    window_size:   int   = BEST_WINDOW_SIZE,
    hidden_size:   int   = BEST_HIDDEN_SIZE,
    num_layers:    int   = BEST_NUM_LAYERS,
    lr:            float = BEST_LR,
    num_epochs:    int   = BEST_EPOCHS,
    batch_size:    int   = BEST_BATCH_SIZE,
) -> dict:
    """
    1. Inietta rumore nel training set (Spark)
    2. Raccoglie training+test su driver come numpy, ordinati per timestamp
    3. Salva .npz su filesystem condiviso (leggibile dai worker)
    4. Lancia TorchDistributor → training DDP distribuito sul cluster
    5. Restituisce il dizionario dei risultati
    """
    t0 = time.time()
    os.makedirs(TMP_DATA_DIR, exist_ok=True)

    # ── Iniezione rumore ───────────────────────────────────────────────────
    if percentage == 0.0:
        noisy_df = pt_train.original
    else:
        noisy_df = getattr(pt_train, noise_type)(
            percentage = percentage,
            columns    = feature,
            seed       = seed,
        )

    # Ordina per timestamp — fondamentale per preservare la gerarchia temporale
    noisy_df = noisy_df.orderBy("timestamp").drop("timestamp")
    noisy_df.cache()
    n_train = noisy_df.count() if noise_type == "duplicated" else pt_train.original.count()

    # ── Raccolta su driver — ordinamento temporale già applicato ──────────
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

    # ── Salva .npz su filesystem condiviso ────────────────────────────────
    data_path = os.path.join(TMP_DATA_DIR, f"run_{noise_type}_{seed}.npz")
    np.savez(
        data_path,
        train_features = train_features,
        train_labels   = train_labels,
        test_features  = test_features,
        test_labels    = test_labels,
    )

    # ── Training distribuito via TorchDistributor ─────────────────────────
    # num_processes = 4 → un processo per executor; backend gloo (CPU)
    distributor = TorchDistributor(
        num_processes = 4,
        local_mode    = False,
        use_gpu       = False,
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

    # ── Cleanup ────────────────────────────────────────────────────────────
    try:
        os.remove(data_path)
    except OSError:
        pass
    noisy_df.unpersist()

    return {
        "noise_type":    noise_type,
        "feature":       feature_label,
        "percentage":    percentage,
        "seed":          seed,
        "f1":            metrics["f1"],
        "auc":           metrics["auc"],
        "n_train":       n_train,
        "duration_s":    time.time() - t0,
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
    """
    Esegue tutti i 2.000 run sperimentali.
    I parametri possono essere passati esplicitamente oppure vengono letti
    dalle variabili globali BEST_* impostate in cima al file.
    Scrittura incrementale su RESULTS_PATH — il loop è riprendibile.
    """
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

    # ── Definizione dei run ────────────────────────────────────────────────
    # Il rumore viene iniettato su NOISE_FEATURES (3 colonne)
    feature_configs = [
        ("NOISE_FEATURES (DV_pressure+Oil_temperature+TP3)", NOISE_FEATURES),
    ]

    all_runs = [
        (nt, fl, fa, lvl, s)
        for nt          in NOISE_TYPES
        for (fl, fa)    in feature_configs
        for lvl         in NOISE_LEVELS
        for s           in range(N_RUNS)
    ]

    # ── Resume ────────────────────────────────────────────────────────────
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

    runs_to_do = [r for r in all_runs if (r[0], r[1], r[3], r[4]) not in done]
    print(f"Run da eseguire : {len(runs_to_do)} / {len(all_runs)}\n")

    # ── Signal handler ─────────────────────────────────────────────────────
    def _sig(sig, frame):
        print("\n\n⚠️  Interruzione — uscita pulita.")
        sys.exit(0)
    signal.signal(signal.SIGINT,  _sig)
    signal.signal(signal.SIGTERM, _sig)

    # ── Loop principale ────────────────────────────────────────────────────
    with open(RESULTS_PATH, "a") as out_f:
        for i, (noise_type, feature_label, feature_arg, percentage, seed) in enumerate(runs_to_do):
            try:
                result = run_experiment(
                    pt_train      = pt_train,
                    base_test_df  = base_test_df,
                    noise_type    = noise_type,
                    feature       = feature_arg,
                    feature_label = feature_label,
                    percentage    = percentage,
                    seed          = seed,
                    window_size   = window_size,
                    hidden_size   = hidden_size,
                    num_layers    = num_layers,
                    lr            = lr,
                    num_epochs    = num_epochs,
                    batch_size    = batch_size,
                )
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
                print(f"\n❌ ERRORE ({noise_type}, {feature_label}, {percentage:.0%}, seed={seed}):")
                traceback.print_exc()
                print("— continuo con il prossimo run —\n")
                continue

    print(f"\n✅ Loop completato — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"✅ Risultati in: {RESULTS_PATH}")

    # ── Aggregazione finale ────────────────────────────────────────────────
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