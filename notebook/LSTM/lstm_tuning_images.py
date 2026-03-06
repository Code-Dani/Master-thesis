import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
import numpy as np

# ── Dati ──────────────────────────────────────────────────────────────────
data = [
    (1,  30, "[64]",       1, "1e-3",  5, 0.9628, 0.9409, 147.3),
    (2,  30, "[64]",       1, "1e-3", 10, 0.9628, 0.9388, 282.0),
    (3,  30, "[64]",       1, "1e-4",  5, 0.9632, 0.9826, 142.5),
    (4,  30, "[64]",       1, "1e-4", 10, 0.9629, 0.9787, 284.0),
    (5,  30, "[64]",       2, "1e-3",  5, 0.9628, 0.9414, 342.9),
    (6,  30, "[64]",       2, "1e-3", 10, 0.9632, 0.9933, 687.3),
    (7,  30, "[64]",       2, "1e-4",  5, 0.9630, 0.9827, 340.9),
    (8,  30, "[64]",       2, "1e-4", 10, 0.9627, 0.9467, 672.6),
    (9,  30, "[128]",      1, "1e-3",  5, 0.9629, 0.9360, 328.3),
    (10, 30, "[128]",      1, "1e-3", 10, 0.9629, 0.9893, 649.3),
    (11, 30, "[128]",      1, "1e-4",  5, 0.9629, 0.9693, 327.6),
    (12, 30, "[128]",      1, "1e-4", 10, 0.9628, 0.9519, 649.2),
    (13, 30, "[128]",      2, "1e-3",  5, 0.9631, 0.9399, 879.5),
    (14, 30, "[128]",      2, "1e-3", 10, 0.9630, 0.9417,1747.0),
    (15, 30, "[128]",      2, "1e-4",  5, 0.9632, 0.9410, 882.0),
    (16, 30, "[128]",      2, "1e-4", 10, 0.9629, 0.9403,1745.9),
    (17, 30, "[256]",      1, "1e-3",  5, 0.9627, 0.9481, 966.5),
    (18, 30, "[256]",      1, "1e-3", 10, 0.9630, 0.9409,1915.3),
    (19, 30, "[256]",      1, "1e-4",  5, 0.9628, 0.9438, 966.9),
    (20, 30, "[256]",      1, "1e-4", 10, 0.9630, 0.9426,1915.5),
    (21, 30, "[256]",      2, "1e-3",  5, 0.9627, 0.9594,2749.4),
    (22, 30, "[256]",      2, "1e-3", 10, 0.9443, 0.9943,5464.5),
    (23, 30, "[256]",      2, "1e-4",  5, 0.9631, 0.9408,2755.8),
    (24, 30, "[256]",      2, "1e-4", 10, 0.9633, 0.9406,5492.3),
    (25, 30, "[128,64]",   1, "1e-3",  5, 0.9627, 0.9630, 337.1),
    (26, 30, "[128,64]",   1, "1e-3", 10, 0.9631, 0.9695, 665.4),
    (27, 30, "[128,64]",   1, "1e-4",  5, 0.9632, 0.9652, 336.6),
    (28, 30, "[128,64]",   1, "1e-4", 10, 0.9631, 0.9464, 663.8),
    (29, 30, "[128,64]",   2, "1e-3",  5, 0.9624, 0.9915, 893.0),
    (30, 30, "[128,64]",   2, "1e-3", 10, 0.9496, 0.9391,1770.5),
]

idx    = [d[0]   for d in data]
archs  = [d[2]   for d in data]
layers = [d[3]   for d in data]
lrs    = [d[4]   for d in data]
epochs = [d[5]   for d in data]
f1s    = [d[6]   for d in data]
aucs   = [d[7]   for d in data]
times  = [d[8]   for d in data]

best_idx = aucs.index(max(aucs))  # combo 6 (idx=5)

# ── Colori per architettura ───────────────────────────────────────────────
arch_colors = {
    "[64]":     "#4C72B0",
    "[128]":    "#DD8452",
    "[256]":    "#55A868",
    "[128,64]": "#C44E52",
}
colors = [arch_colors[a] for a in archs]

# ── Marker per lstm_layers ────────────────────────────────────────────────
markers = {1: "o", 2: "^"}

# =============================================================================
# FIGURA 1 — F1 e AUC per combinazione
# =============================================================================
fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
fig.suptitle("LSTM Hyperparameter Tuning — window\\_size = 30s", fontsize=14, fontweight="bold")

x = np.arange(len(idx))

# ── F1 ────────────────────────────────────────────────────────────────────
ax1 = axes[0]
for i, (xi, f1, c, lay) in enumerate(zip(x, f1s, colors, layers)):
    ax1.scatter(xi, f1, color=c, marker=markers[lay], s=60, zorder=3)
ax1.axhline(y=max(f1s), color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
ax1.scatter(x[best_idx], f1s[best_idx], color="gold", marker="*", s=250,
            zorder=5, edgecolors="black", linewidths=0.8, label="Best AUC")
ax1.set_ylabel("F1-score (weighted)", fontsize=11)
ax1.set_ylim(0.940, 0.966)
ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.4f}"))
ax1.grid(axis="y", alpha=0.3)
ax1.legend(fontsize=10)

# ── AUC ───────────────────────────────────────────────────────────────────
ax2 = axes[1]
for i, (xi, auc, c, lay) in enumerate(zip(x, aucs, colors, layers)):
    ax2.scatter(xi, auc, color=c, marker=markers[lay], s=60, zorder=3)
ax2.scatter(x[best_idx], aucs[best_idx], color="gold", marker="*", s=250,
            zorder=5, edgecolors="black", linewidths=0.8, label=f"Best: #{idx[best_idx]}  AUC={aucs[best_idx]:.4f}")
ax2.set_ylabel("AUC-ROC", fontsize=11)
ax2.set_xlabel("Combinazione #", fontsize=11)
ax2.set_xticks(x)
ax2.set_xticklabels([str(i) for i in idx], fontsize=7, rotation=45)
ax2.set_ylim(0.930, 1.000)
ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.4f}"))
ax2.grid(axis="y", alpha=0.3)
ax2.legend(fontsize=10)

# Legenda architettura
arch_patches = [mpatches.Patch(color=c, label=a) for a, c in arch_colors.items()]
layer_handles = [
    plt.scatter([], [], marker="o", color="gray", s=60, label="lstm\\_layers=1"),
    plt.scatter([], [], marker="^", color="gray", s=60, label="lstm\\_layers=2"),
]
fig.legend(handles=arch_patches + layer_handles,
           loc="lower center", ncol=6, fontsize=9,
           bbox_to_anchor=(0.5, -0.04), frameon=True)

plt.tight_layout(rect=(0, 0.04, 1, 1))
plt.savefig("lstm_tuning_metrics.png", dpi=150, bbox_inches="tight")
plt.close()
print("Salvato: lstm_tuning_metrics.png")

# =============================================================================
# FIGURA 2 — AUC vs Tempo (efficienza)
# =============================================================================
fig2, ax = plt.subplots(figsize=(12, 6))
ax.set_title("LSTM Tuning — AUC vs Tempo di training", fontsize=13, fontweight="bold")

for i, (t, auc, c, lay, ep) in enumerate(zip(times, aucs, colors, layers, epochs)):
    ax.scatter(t, auc, color=c, marker=markers[lay], s=80, zorder=3, alpha=0.85)

# Annota le migliori AUC
top = sorted(enumerate(aucs), key=lambda x: x[1], reverse=True)[:5]
for rank, (i, auc) in enumerate(top):
    ax.annotate(f"#{idx[i]}\nAUC={auc:.4f}",
                xy=(times[i], auc),
                xytext=(times[i] + 80, auc - 0.004 * (rank % 2 * 2 - 1)),
                fontsize=7.5, color="black",
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

# Stella sul best
ax.scatter(times[best_idx], aucs[best_idx], color="gold", marker="*",
           s=300, zorder=5, edgecolors="black", linewidths=0.8)

ax.set_xlabel("Tempo di training (s)", fontsize=11)
ax.set_ylabel("AUC-ROC", fontsize=11)
ax.set_ylim(0.930, 1.000)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.4f}"))
ax.grid(alpha=0.3)

arch_patches = [mpatches.Patch(color=c, label=a) for a, c in arch_colors.items()]
layer_handles = [
    plt.scatter([], [], marker="o", color="gray", s=60, label="lstm\\_layers=1"),
    plt.scatter([], [], marker="^", color="gray", s=60, label="lstm\\_layers=2"),
]
ax.legend(handles=arch_patches + layer_handles, fontsize=9, loc="lower right")

plt.tight_layout()
plt.savefig("lstm_tuning_efficiency.png", dpi=150, bbox_inches="tight")
plt.close()
print("Salvato: lstm_tuning_efficiency.png")