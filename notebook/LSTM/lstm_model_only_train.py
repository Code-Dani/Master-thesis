"""
train_lstm.py
=============
FASE 2 — Legge i parquet pre-generati e fa il training LSTM.
Niente Spark: puro pandas + PyTorch.

Le run sono ordinate per parquet: 5 seed consecutivi sullo stesso file,
poi si passa al successivo → UN solo dataset in memoria alla volta.

Esecuzione:
    python train_lstm.py
"""

# =============================================================================
# 🗺️ PATH — MODIFICA QUI
# =============================================================================
NOISY_DIR    = r"D:\Users\satri\Pictures\github\Deep-Learning-Robustness-Study\notebook\Dataset\NEW\noisy"
RESULTS_PATH = r"D:\Users\satri\Pictures\github\Deep-Learning-Robustness-Study\notebook\LSTM\lstm_results_final.jsonl"


# =============================================================================
import os, json, time, signal, sys
from datetime import datetime

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, roc_auc_score

torch.set_num_threads(os.cpu_count() or 1)

# =============================================================================
# IPERPARAMETRI (identici al notebook originale)
# =============================================================================
BEST_WINDOW_SIZE = 30
BEST_HIDDEN_SIZE = 64
BEST_NUM_LAYERS  = 2
BEST_LR          = 1e-3
BEST_EPOCHS      = 10
BEST_BATCH_SIZE  = 512

ALL_FEATURES = [
    "TP2_scaled", "TP3_scaled", "H1_scaled", "DV_pressure_scaled",
    "Reservoirs_scaled", "Oil_temperature_scaled", "Motor_current_scaled",
    "COMP", "DV_eletric", "Towers", "MPG", "LPS", "Pressure_switch",
]
LABEL_COL  = "target"
N_FEATURES = len(ALL_FEATURES)
N_RUNS     = 20
SEEDS      = list(range(N_RUNS))

NOISE_FEATURES = ["DV_pressure_scaled", "Oil_temperature_scaled", "TP3_scaled"]
NOISE_LEVELS   = [0.0, 0.1, 0.2, 0.3, 0.5]

# =============================================================================
# LSTM
# =============================================================================
class _WinDS(Dataset):
    def __init__(self, feats: np.ndarray, labs: np.ndarray, ws: int):
        self.f  = torch.tensor(feats, dtype=torch.float32)
        self.l  = torch.tensor(labs,  dtype=torch.long)
        self.ws = ws
    def __len__(self):          return len(self.f) - self.ws + 1
    def __getitem__(self, i):   return self.f[i:i+self.ws], self.l[i+self.ws-1]

class _LSTM(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features, hidden_size, num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, 2)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])

def train_and_eval(
    train_features: np.ndarray, train_labels: np.ndarray,
    test_features:  np.ndarray, test_labels:  np.ndarray,
    seed: int,
    run_label: str = ""
) -> tuple[float, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds     = _WinDS(train_features, train_labels, BEST_WINDOW_SIZE)
    test_ds      = _WinDS(test_features,  test_labels,  BEST_WINDOW_SIZE)
    train_loader = DataLoader(train_ds, batch_size=BEST_BATCH_SIZE, shuffle=True,
                              num_workers=6, persistent_workers=True, prefetch_factor=2)
    test_loader  = DataLoader(test_ds,  batch_size=BEST_BATCH_SIZE, shuffle=False,
                              num_workers=6, persistent_workers=True, prefetch_factor=2)

    model     = _LSTM(N_FEATURES, BEST_HIDDEN_SIZE, BEST_NUM_LAYERS).to(device)
    n_pos     = max(int((train_labels == 1).sum()), 1)
    n_neg     = int((train_labels == 0).sum())
    weight    = torch.tensor([1.0, n_neg / n_pos], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=BEST_LR)

    # --- Training con log per epoch ---
    model.train()
    t_train_start = time.time()
    for epoch in range(BEST_EPOCHS):
        epoch_loss = 0.0
        t_epoch = time.time()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss   = epoch_loss / len(train_loader)
        epoch_time = time.time() - t_epoch
        eta        = (BEST_EPOCHS - epoch - 1) * epoch_time
        print(f"      epoch {epoch+1:>2}/{BEST_EPOCHS}  "
              f"loss={avg_loss:.4f}  "
              f"({epoch_time:.1f}s/epoch  ETA training: {eta:.0f}s)",
              flush=True)

    # --- Evaluation ---
    model.eval()
    preds, probs, labs = [], [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            preds.extend(logits.argmax(dim=1).cpu().numpy())
            labs.extend(yb.cpu().numpy())

    f1  = f1_score(np.array(labs), np.array(preds), average="weighted", zero_division=0)
    auc = roc_auc_score(np.array(labs), np.array(probs)) if len(np.unique(labs)) > 1 else 0.0
    return float(f1), float(auc)

# =============================================================================
# I/O HELPERS
# =============================================================================
def load_parquet(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Carica un parquet, ordina per timestamp, imputa NaN con media colonna."""
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)
    col_means = df[ALL_FEATURES].mean()
    df[ALL_FEATURES] = df[ALL_FEATURES].fillna(col_means)
    features = np.array(df[ALL_FEATURES], dtype=np.float32)
    labels   = np.array(df[LABEL_COL],   dtype=np.int64)
    return features, labels

def pq_path(noise_type: str, feature_label: str, percentage: float) -> str:
    """Ricostruisce il path del parquet dal nome convenzione."""
    if percentage == 0.0:
        return os.path.join(NOISY_DIR, "baseline")
    pct_str = str(int(round(percentage * 100)))
    return os.path.join(NOISY_DIR, f"{noise_type}_{feature_label}_{pct_str}pct")

# =============================================================================
# PIANO SPERIMENTALE
# =============================================================================
def build_run_list() -> list[dict]:
    combos = []

    for noise_type in ["duplicated", "labels"]:
        for pct in NOISE_LEVELS:
            combos.append({
                "noise_type":   noise_type,
                "feature":      "all",
                "percentage":   pct,
                "parquet_path": pq_path(noise_type, "all", pct),
            })

    for noise_type in ["missing", "noise", "outliers"]:
        for feat in NOISE_FEATURES:
            for pct in NOISE_LEVELS:
                combos.append({
                    "noise_type":   noise_type,
                    "feature":      feat,
                    "percentage":   pct,
                    "parquet_path": pq_path(noise_type, feat, pct),
                })

    runs = []
    for combo in combos:
        for seed in SEEDS:
            runs.append({**combo, "seed": seed})

    runs.sort(key=lambda r: (r["parquet_path"], r["seed"]))
    return runs

# =============================================================================
# MAIN
# =============================================================================
def main():
    test_path = os.path.join(NOISY_DIR, "test_set")
    if not os.path.exists(test_path):
        print(f"❌ Test set non trovato: {test_path}")
        sys.exit(1)

    print("📥 Caricamento test set...")
    test_features, test_labels = load_parquet(test_path)
    print(f"   Test: {len(test_features):,} campioni  "
          f"(fault={int((test_labels==1).sum()):,} / {len(test_labels):,})")

    completed: set[tuple] = set()
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    completed.add((r["noise_type"], r["feature"], r["percentage"], r["seed"]))
                except Exception:
                    pass
        print(f"✅ Run già completati: {len(completed)}")

    all_runs  = build_run_list()
    todo_runs = [r for r in all_runs
                 if (r["noise_type"], r["feature"], r["percentage"], r["seed"]) not in completed]

    total_expected = len(all_runs)
    print(f"\n{'='*65}")
    print(f"TRAINING LSTM — {total_expected} run totali | {len(todo_runs)} rimanenti")
    print(f"Stima iniziale: {len(todo_runs) * 2 / 60:.1f} ore (aggiornata dinamicamente)")
    print(f"{'='*65}\n")

    if len(todo_runs) == 0:
        print("✅ Tutti i run già completati.")
        return

    def _sig(sig, frame):
        print("\n⚠️  Interruzione — risultati salvati fino a qui.")
        sys.exit(0)
    signal.signal(signal.SIGINT,  _sig)
    signal.signal(signal.SIGTERM, _sig)

    current_pq_path: str | None = None
    train_features:  np.ndarray | None = None
    train_labels:    np.ndarray | None = None
    run_times:       list[float] = []

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

    with open(RESULTS_PATH, "a") as out_file:
        for i, run in enumerate(todo_runs):
            pq = run["parquet_path"]

            if pq != current_pq_path:
                if not os.path.exists(pq):
                    print(f"  ⚠️  Parquet non trovato (skip): {pq}")
                    continue
                print(f"\n  📂 Caricamento: {os.path.basename(pq)}")
                train_features, train_labels = load_parquet(pq)
                current_pq_path = pq
                print(f"     {len(train_features):,} campioni  "
                      f"(fault={int((train_labels==1).sum()):,})")

            run_label = (f"{run['noise_type']} | {run['feature']} | "
                         f"{int(run['percentage']*100)}% | seed={run['seed']}")
            print(f"\n  ▶ [{i+1:>4}/{len(todo_runs)}] {run_label}", flush=True)

            t0 = time.time()
            try:
                assert train_features is not None and train_labels is not None, \
                    "train_features/train_labels non caricati — parquet mancante?"
                f1, auc = train_and_eval(
                    train_features, train_labels,
                    test_features,  test_labels,
                    seed=run["seed"],
                    run_label=run_label,
                )
                elapsed = time.time() - t0
                run_times.append(elapsed)

                # ETA dinamica basata sulla media degli ultimi 5 run
                avg_run_time = sum(run_times[-5:]) / len(run_times[-5:])
                remaining    = len(todo_runs) - (i + 1)
                eta_h        = remaining * avg_run_time / 3600

                result = {
                    "noise_type": run["noise_type"],
                    "feature":    run["feature"],
                    "percentage": run["percentage"],
                    "seed":       run["seed"],
                    "f1":         f1,
                    "auc":        auc,
                    "duration_s": round(elapsed, 2),
                    "timestamp":  datetime.now().isoformat(),
                }
                out_file.write(json.dumps(result) + "\n")
                out_file.flush()

                print(f"  ✅ F1={f1:.4f}  AUC={auc:.4f}  "
                      f"({elapsed:.1f}s)  |  ETA: {eta_h:.1f}h ({remaining} run rimanenti)",
                      flush=True)

            except Exception as e:
                print(f"  ❌ ERRORE: {e}", flush=True)
                continue

    print(f"\n✅ COMPLETATO → {RESULTS_PATH}")

if __name__ == "__main__":
    main()