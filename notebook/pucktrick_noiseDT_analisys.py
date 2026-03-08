"""
Script per generare le figure dell'analisi del dataset corrotto.
Da eseguire sulla macchina locale dove sono presenti i file Parquet.

Output: immagini PNG salvate in ./notebook/corruption_figures/
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
from pathlib import Path

# ── Configurazione ────────────────────────────────────────────────────────────
BASE_PATH        = Path(r"D:\Users\satri\Pictures\github\Deep-Learning-Robustness-Study\notebook\Dataset\NEW\noisy")
NEW_BASE_PATH    = Path(r"D:\Users\satri\Pictures\github\Deep-Learning-Robustness-Study\notebook\Dataset\NEW\noisy_new")
CLEAN_TRAIN_PATH = Path(r"D:\Users\satri\Pictures\github\Deep-Learning-Robustness-Study\notebook\Dataset\NEW\noisy\baseline")
OUTPUT_DIR       = Path(r"./notebook/corruption_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = Path(r"D:\Users\satri\Pictures\github\Deep-Learning-Robustness-Study\notebook\Dataset\NEW\noisy\generation_log.jsonl")

PERCENTAGES = [0.1, 0.2, 0.3, 0.5]
PCT_LABELS  = ["10%", "20%", "30%", "50%"]
FEATURES    = ["DV_pressure_scaled", "Oil_temperature_scaled", "TP3_scaled"]
FEAT_LABELS = {
    "DV_pressure_scaled":    "DV_pressure",
    "Oil_temperature_scaled":"Oil_temperature",
    "TP3_scaled":            "TP3",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 150,
})


# ── Helper ────────────────────────────────────────────────────────────────────
def load_parquet(folder: Path) -> pd.DataFrame:
    if folder.is_dir():
        files = list(folder.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"Nessun .parquet in {folder}")
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return pd.read_parquet(folder)


def load_log(path: Path):
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "stdout" in obj:
                for sub in obj["stdout"].strip().split("\n"):
                    sub = sub.strip()
                    if sub:
                        entries.append(json.loads(sub))
            else:
                entries.append(obj)
    return entries


# ── 1. Tempi di generazione ───────────────────────────────────────────────────
def plot_generation_times(entries):
    from collections import defaultdict
    by_type = defaultdict(float)
    counts  = defaultdict(int)
    for e in entries:
        key = "noisy" if e["noise_type"] == "noise" else e["noise_type"]
        by_type[key] += e["duration_s"]
        counts[key]  += 1

    order = ["duplicated", "labels", "missing", "noisy", "outliers"]
    times = [by_type[k] for k in order]
    ns    = [counts[k]  for k in order]
    total = sum(by_type.values())

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(order, times, color="#4878d0", edgecolor="white", width=0.55)
    for bar, t, n in zip(bars, times, ns):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                f"{t:.1f}s\n(n={n})", ha="center", va="bottom", fontsize=9.5)
    ax.set_ylabel("Tempo totale (s)")
    ax.set_title(f"Tempo di generazione dataset corrotti per tipo di rumore\n"
                 f"(totale: {total:.1f}s \u2248 {total/60:.1f} min, 44 dataset)")
    ax.set_ylim(0, max(times) * 1.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = OUTPUT_DIR / "generation_times.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"\u2713 Salvato: {out}")


# ── 2. Distribuzione target – rumore labels ───────────────────────────────────
def plot_labels_distribution(clean_df):
    n_total = len(clean_df)

    fig, axes = plt.subplots(1, 5, figsize=(14, 4), sharey=True)

    def draw_bar(ax, n0, n1, title):
        total = n0 + n1
        ax.bar(["Classe 0\n(no guasto)", "Classe 1\n(guasto)"],
               [n0, n1], color=["#4878d0", "#d62728"], edgecolor="white", width=0.55)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, n_total * 1.25)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for i, v in enumerate([n0, n1]):
            pct = v / total * 100
            ax.text(i, v + n_total * 0.01, f"{pct:.1f}%", ha="center", fontsize=9)

    draw_bar(axes[0], (clean_df["target"] == 0).sum(),
             (clean_df["target"] == 1).sum(), "0% (baseline)")
    for ax, pct, lbl in zip(axes[1:], PERCENTAGES, PCT_LABELS):
        folder = BASE_PATH / f"labels_all_{int(pct*100)}pct"
        try:
            df = load_parquet(folder)
            draw_bar(ax, (df["target"] == 0).sum(), (df["target"] == 1).sum(), lbl)
        except Exception as ex:
            ax.set_title(f"{lbl}\n(errore: {ex})")

    fig.suptitle("Distribuzione della classe target dopo il rumore labels", fontsize=12, y=1.02)
    axes[0].set_ylabel("Numero di campioni")
    plt.tight_layout()
    out = OUTPUT_DIR / "labels_target_distribution.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"\u2713 Salvato: {out}")


# ── 3a. Pie chart NaN – rumore missing (griglia 3x5) ─────────────────────────
def plot_missing_nan_piecharts(clean_df):
    all_pcts   = [0.0] + PERCENTAGES
    all_labels = ["0%\n(baseline)"] + PCT_LABELS

    fig, axes = plt.subplots(3, 5, figsize=(14, 9))

    for row, feat in enumerate(FEATURES):
        feat_name = FEAT_LABELS[feat]
        for col, (pct, lbl) in enumerate(zip(all_pcts, all_labels)):
            ax = axes[row, col]
            if pct == 0.0:
                series = clean_df[feat]
            else:
                folder = BASE_PATH / f"missing_{feat}_{int(pct*100)}pct"
                try:
                    series = load_parquet(folder)[feat]
                except Exception as ex:
                    ax.set_title(f"{lbl}\n(errore)")
                    print(f"  Attenzione missing {feat} {lbl}: {ex}")
                    continue

            n_total   = len(series)
            n_nan     = int(series.isna().sum())
            n_present = n_total - n_nan

            _, _, autotexts = ax.pie(
                [n_present, n_nan],
                labels=["Presenti", "NaN"],
                colors=["#4878d0", "#d62728"],
                autopct="%1.1f%%",
                startangle=90,
                wedgeprops={"edgecolor": "white", "linewidth": 1.0},
                textprops={"fontsize": 7},
            )
            for at in autotexts:
                at.set_fontsize(6.5)

            if row == 0:
                ax.set_title(f"{lbl}\n({n_nan:,} NaN)", fontsize=8.5, pad=4)
            else:
                ax.set_title(f"({n_nan:,} NaN)", fontsize=8, pad=4)

        axes[row, 0].set_ylabel(feat_name, fontsize=9, labelpad=8)

    fig.suptitle(
        "Proporzione di NaN introdotti dal rumore missing per feature e percentuale",
        fontsize=11, y=1.01
    )
    plt.tight_layout()
    out = OUTPUT_DIR / "missing_nan_pie_all.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"\u2713 Salvato: {out}")


# ── 3b. Istogrammi – noisy e outliers ────────────────────────────────────────
def plot_feature_distributions(clean_df):
    methods_map = {
        "noisy":    ("noise",    "#2ca02c"),
        "outliers": ("outliers", "#ff7f0e"),
    }
    for feat in FEATURES:
        feat_name = FEAT_LABELS[feat]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        col_clean = clean_df[feat].dropna()

        for ax, (method_label, (noise_key, color)) in zip(axes, methods_map.items()):
            folder = BASE_PATH / f"{noise_key}_{feat}_50pct"
            try:
                col50 = load_parquet(folder)[feat].dropna()
                bins  = np.linspace(min(col_clean.min(), col50.min()),
                                    max(col_clean.max(), col50.max()), 60)
                ax.hist(col_clean, bins=bins, density=True,
                        alpha=0.55, color="#4878d0", label="0% (pulito)")
                ax.hist(col50,     bins=bins, density=True,
                        alpha=0.55, color=color,    label="50% corrotto")
                ax.set_title(method_label)
            except Exception as ex:
                ax.set_title(f"{method_label}\n(errore)")
                print(f"  Attenzione: {feat} / {method_label}: {ex}")

            ax.set_xlabel("Valore (scalato)")
            ax.set_ylabel("Densit\u00e0")
            ax.legend(fontsize=9)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        fig.suptitle(
            f"Distribuzione di {feat_name}: pulito vs 50% di rumore",
            fontsize=12, y=1.02
        )
        plt.tight_layout()
        out = OUTPUT_DIR / f"feature_dist_{feat}.png"
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        print(f"\u2713 Salvato: {out}")


# ── 4. Duplicated: barre impilate + crescita guasti ──────────────────────────
def plot_duplicated_analysis(clean_df):
    n_clean = len(clean_df)
    n0_base = int((clean_df["target"] == 0).sum())
    n1_base = int((clean_df["target"] == 1).sum())

    sizes       = [n_clean]
    size_labels = ["0%\n(base)"]
    n0_all = [n0_base]
    n1_all = [n1_base]
    n0_t1  = [n0_base]
    n1_t1  = [n1_base]

    for pct, lbl in zip(PERCENTAGES, PCT_LABELS):
        folder_all = BASE_PATH / f"duplicated_all_{int(pct*100)}pct"
        try:
            df_all = load_parquet(folder_all)
            sizes.append(len(df_all))
            n0_all.append(int((df_all["target"] == 0).sum()))
            n1_all.append(int((df_all["target"] == 1).sum()))
        except Exception as ex:
            sizes.append(None)
            n0_all.append(None)
            n1_all.append(None)
            print(f"  Attenzione duplicated_all {lbl}: {ex}")
        size_labels.append(lbl)

        folder_t1 = NEW_BASE_PATH / f"duplicated_target1_{int(pct*100)}pct"
        try:
            df_t1 = load_parquet(folder_t1)
            n0_t1.append(int((df_t1["target"] == 0).sum()))
            n1_t1.append(int((df_t1["target"] == 1).sum()))
        except Exception as ex:
            n0_t1.append(None)
            n1_t1.append(None)
            print(f"  Attenzione duplicated_target1 {lbl}: {ex}")

    pct_labels_full = ["0%\n(base)", "10%", "20%", "30%", "50%"]

    from matplotlib.patches import Patch
    fig, (ax_size, ax_all_stacked, ax_t1) = plt.subplots(1, 3, figsize=(16, 5))

    # --- Sinistra: barre impilate classe 0/1 per duplicated casuale ---
    xs_s = np.arange(len(pct_labels_full))
    for i, (n0, n1, x) in enumerate(zip(n0_all, n1_all, xs_s)):
        if n0 is None or n1 is None:
            continue
        total = n0 + n1
        ax_size.bar(x, n0, color="#4878d0", edgecolor="white", width=0.55)
        ax_size.bar(x, n1, bottom=n0, color="#d62728", edgecolor="white", width=0.55)
        delta = f"\n(+{(total - n_clean)/n_clean*100:.0f}%)" if total != n_clean else ""
        ax_size.text(x, total + n_clean * 0.01,
                     f"{total:,}{delta}", ha="center", fontsize=7.5)
        if n0 / total * 100 > 5:
            ax_size.text(x, n0 / 2, f"{n0/total*100:.1f}%",
                         ha="center", va="center", fontsize=7.5,
                         color="white", fontweight="bold")
        if n1 / total * 100 > 0.5:
            ax_size.text(x, n0 + n1 / 2, f"{n1/total*100:.2f}%",
                         ha="center", va="center", fontsize=7,
                         color="white", fontweight="bold")
    ax_size.set_xticks(xs_s)
    ax_size.set_xticklabels(pct_labels_full, fontsize=9)
    ax_size.set_ylabel("Numero di campioni")
    ax_size.set_title("Dimensione train set per classe\n(duplicated senza condizione nei filtri)")
    ax_size.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax_size.spines["top"].set_visible(False)
    ax_size.spines["right"].set_visible(False)
    ax_size.legend(handles=[Patch(color="#4878d0", label="Classe 0 (no guasto)"),
                             Patch(color="#d62728", label="Classe 1 (guasto)")],
                   fontsize=8, loc="upper left")

    # --- Centro: barre impilate classe 0/1 per duplicated target=1 ---
    for i, (n0, n1, x) in enumerate(zip(n0_t1, n1_t1, xs_s)):
        if n0 is None or n1 is None:
            continue
        total = n0 + n1
        ax_all_stacked.bar(x, n0, color="#4878d0", edgecolor="white", width=0.55)
        ax_all_stacked.bar(x, n1, bottom=n0, color="#d62728", edgecolor="white", width=0.55)
        delta = f"\n(+{(total - n_clean)/n_clean*100:.0f}%)" if total != n_clean else ""
        ax_all_stacked.text(x, total + n_clean * 0.01,
                            f"{total:,}{delta}", ha="center", fontsize=7.5)
        if n0 / total * 100 > 5:
            ax_all_stacked.text(x, n0 / 2, f"{n0/total*100:.1f}%",
                                ha="center", va="center", fontsize=7.5,
                                color="white", fontweight="bold")
        if n1 / total * 100 > 0.5:
            ax_all_stacked.text(x, n0 + n1 / 2, f"{n1/total*100:.2f}%",
                                ha="center", va="center", fontsize=7,
                                color="white", fontweight="bold")
    ax_all_stacked.set_xticks(xs_s)
    ax_all_stacked.set_xticklabels(pct_labels_full, fontsize=9)
    ax_all_stacked.set_ylabel("Numero di campioni")
    ax_all_stacked.set_title("Dimensione train set per classe\n(duplicated su target=1)")
    ax_all_stacked.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax_all_stacked.spines["top"].set_visible(False)
    ax_all_stacked.spines["right"].set_visible(False)
    ax_all_stacked.legend(handles=[Patch(color="#4878d0", label="Classe 0 (no guasto)"),
                                    Patch(color="#d62728", label="Classe 1 (guasto)")],
                          fontsize=8, loc="upper left")

    # --- Destra: valori assoluti classe 1 per duplicated target=1 ---
    valid = [(i, v) for i, v in enumerate(n1_t1) if v is not None]
    xs_v  = [x for x, _ in valid]
    ys_v  = [v for _, v in valid]
    bar_colors_t1 = ["#4878d0"] + ["#d62728"] * (len(xs_v) - 1)
    bars_t1 = ax_t1.bar(xs_v, ys_v, color=bar_colors_t1, edgecolor="white", width=0.55)
    for bar, y, i in zip(bars_t1, ys_v, xs_v):
        total = n0_t1[i] + n1_t1[i]
        pct   = y / total * 100
        ax_t1.text(bar.get_x() + bar.get_width() / 2,
                   y + max(ys_v) * 0.02,
                   f"{y:,}\n({pct:.2f}%)", ha="center", fontsize=8)
    ax_t1.set_xticks(xs_v)
    ax_t1.set_xticklabels([pct_labels_full[i] for i in xs_v], fontsize=9)
    ax_t1.set_ylabel("Campioni Classe 1 (guasto)")
    ax_t1.set_title("Crescita assoluta dei guasti\n(con condizione target=1)")
    ax_t1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax_t1.spines["top"].set_visible(False)
    ax_t1.spines["right"].set_visible(False)

    fig.suptitle("Analisi del rumore duplicated: dimensione e distribuzione del target",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    out = OUTPUT_DIR / "duplicated_analysis.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"\u2713 Salvato: {out}")


# ── 5. TP3+Reservoirs: griglia 2x2 (righe: feature, colonne: metodo) ─────────
def plot_tp3_reservoir_distributions(clean_df):
    """
    Figura 1: pie chart NaN — griglia 2x5 (TP3 e Reservoirs per 5 percentuali)
    Figura 2: istogrammi  — griglia 2x2 (TP3/Reservoirs x noisy/outliers) a 50%
    """
    TP3_RES_FEATURES = [
        ("TP3_scaled",        "TP3"),
        ("Reservoirs_scaled", "Reservoirs"),
    ]
    noise_methods = [
        ("noisy",    "noise",    "#2ca02c"),
        ("outliers", "outliers", "#ff7f0e"),
    ]
    all_pcts   = [0.0] + PERCENTAGES
    all_labels = ["0%\n(baseline)"] + PCT_LABELS

    # ── Figura 1: pie chart NaN — griglia 2x5 ────────────────────────────────
    fig, axes = plt.subplots(2, 5, figsize=(14, 7))

    for row, (feat_col, feat_name) in enumerate(TP3_RES_FEATURES):
        for col, (pct, lbl) in enumerate(zip(all_pcts, all_labels)):
            ax = axes[row, col]
            if pct == 0.0:
                series = clean_df[feat_col] if feat_col in clean_df.columns else None
            else:
                folder = NEW_BASE_PATH / f"missing_TP3_Reservoirs_{int(pct*100)}pct"
                try:
                    df = load_parquet(folder)
                    series = df[feat_col] if feat_col in df.columns else None
                    if series is None:
                        raise KeyError(f"Colonna {feat_col!r} non trovata. Colonne: {list(df.columns)}")
                except Exception as ex:
                    ax.set_title(f"{lbl}\n(errore)")
                    print(f"  Attenzione TP3+Res missing {feat_name} {lbl}: {ex}")
                    continue

            if series is None:
                ax.set_title(f"{lbl}\n(colonna assente)")
                continue

            n_total   = len(series)
            n_nan     = int(series.isna().sum())
            n_present = n_total - n_nan

            _, _, autotexts = ax.pie(
                [n_present, n_nan],
                labels=["Presenti", "NaN"],
                colors=["#4878d0", "#d62728"],
                autopct="%1.1f%%",
                startangle=90,
                wedgeprops={"edgecolor": "white", "linewidth": 1.0},
                textprops={"fontsize": 7},
            )
            for at in autotexts:
                at.set_fontsize(6.5)

            if row == 0:
                ax.set_title(f"{lbl}\n({n_nan:,} NaN)", fontsize=8.5, pad=4)
            else:
                ax.set_title(f"({n_nan:,} NaN)", fontsize=8, pad=4)

        axes[row, 0].set_ylabel(feat_name, fontsize=9, labelpad=8)

    fig.suptitle(
        "Proporzione di NaN introdotti dal rumore missing — TP3 e Reservoirs simultanei\n"
        "(train set, per percentuale di corruzione)",
        fontsize=11, y=1.01
    )
    plt.tight_layout()
    out = OUTPUT_DIR / "tp3_reservoir_missing_nan_pie.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"\u2713 Salvato: {out}")

    # ── Figura 2: istogrammi 2x2 ─────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    for row, (feat_col, feat_name) in enumerate(TP3_RES_FEATURES):
        col_clean = clean_df[feat_col].dropna() if feat_col in clean_df.columns else None

        for col_idx, (method_label, noise_key, color) in enumerate(noise_methods):
            ax = axes[row, col_idx]
            folder = NEW_BASE_PATH / f"{noise_key}_TP3_Reservoirs_50pct"
            try:
                df50 = load_parquet(folder)
                if feat_col not in df50.columns:
                    raise KeyError(f"Colonna {feat_col!r} non trovata. Disponibili: {list(df50.columns)}")
                col50 = df50[feat_col].dropna()
                if col_clean is None:
                    raise KeyError(f"Colonna {feat_col!r} assente nel train set pulito")
                bins = np.linspace(min(col_clean.min(), col50.min()),
                                   max(col_clean.max(), col50.max()), 60)
                ax.hist(col_clean, bins=bins, density=True,
                        alpha=0.55, color="#4878d0", label="0% (pulito)")
                ax.hist(col50,     bins=bins, density=True,
                        alpha=0.55, color=color,    label="50% corrotto")
            except Exception as ex:
                ax.text(0.5, 0.5, str(ex), ha="center", va="center",
                        transform=ax.transAxes, fontsize=7, wrap=True)
                print(f"  Attenzione TP3+Res {feat_name}/{method_label}: {ex}")

            if row == 0:
                ax.set_title(method_label, fontsize=11)
            if col_idx == 0:
                ax.set_ylabel(f"{feat_name}\nDensit\u00e0", fontsize=10)
            else:
                ax.set_ylabel("Densit\u00e0")
            ax.set_xlabel("Valore scalato")
            ax.legend(fontsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Distribuzione di TP3 e Reservoirs con rumore simultaneo: pulito vs 50%\n"
        "(noisy e outliers — missing mostrato separatamente)",
        fontsize=12, y=1.01
    )
    plt.tight_layout()
    out = OUTPUT_DIR / "tp3_reservoir_dist.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"\u2713 Salvato: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Caricamento log di generazione ===")
    entries = load_log(LOG_PATH)
    print(f"  {len(entries)} entry trovate")

    print("\n=== Caricamento train set pulito ===")
    try:
        clean_df = load_parquet(CLEAN_TRAIN_PATH)
        print(f"  Train set: {len(clean_df):,} righe, colonne: {list(clean_df.columns)}")
    except Exception as e:
        print(f"  ERRORE caricamento train set: {e}")
        clean_df = None

    print("\n=== Generazione figure ===")
    plot_generation_times(entries)

    if clean_df is not None:
        plot_labels_distribution(clean_df)
        plot_missing_nan_piecharts(clean_df)
        plot_feature_distributions(clean_df)
        plot_duplicated_analysis(clean_df)
        plot_tp3_reservoir_distributions(clean_df)

    print(f"\nFatto! Figure salvate in: {OUTPUT_DIR.resolve()}")