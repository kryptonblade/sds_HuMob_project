import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Data (ONLY FIRST 2) ───────────────────────────────────────────────────────
series = [
    {
        "name": "Original",
        "color": "#FF6B6B",
        "values": [0.0137, 0.0462, 0.0585, 0.0752, 0.0835, 0.0854,
                   0.0859, 0.0936, 0.0902, 0.0954, 0.0899, 0.0914,
                   0.0932, 0.0889, 0.0880, 0.0902],
    },
    {
        "name": "Improved (+cyclic encoding)",
        "color": "#4ADE80",
        "values": [0.0073, 0.0359, 0.0647, 0.0738, 0.0786, 0.0878,
                   0.0924, 0.0882, 0.0928, 0.0924, 0.0951, 0.0985,
                   0.0940, 0.0919, 0.0914, 0.0947],
    }
]

BASELINE = 0.00052

# ── Style ─────────────────────────────────────────────────────────────────────
BG_DARK  = "#0B0E17"
BG_CARD  = "#131725"
GRID_COL = "#1E2538"
TEXT_PRI = "#E8EAF0"
TEXT_MUT = "#7A82A8"

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "text.color":        TEXT_PRI,
    "axes.labelcolor":   TEXT_MUT,
    "xtick.color":       TEXT_MUT,
    "ytick.color":       TEXT_MUT,
    "axes.facecolor":    BG_CARD,
    "figure.facecolor":  BG_DARK,
    "axes.edgecolor":    GRID_COL,
    "grid.color":        GRID_COL,
    "grid.linestyle":    "--",
    "grid.linewidth":    0.6,
})

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

ax.yaxis.grid(True)
ax.set_axisbelow(True)

# ── Plot ──────────────────────────────────────────────────────────────────────
for s in series:
    epochs = np.arange(1, len(s["values"]) + 1)
    ax.plot(
        epochs, s["values"],
        color=s["color"],
        linewidth=2.5,
        marker="o",
        markersize=5,
        label=s["name"],
    )

# ── Baseline ──────────────────────────────────────────────────────────────────
ax.axhline(BASELINE, color="#FF4444", linestyle="--", linewidth=1.2)
ax.text(16.1, BASELINE + 0.0008, f"baseline {BASELINE}",
        color="#FF4444", fontsize=9)

# ── Axes ──────────────────────────────────────────────────────────────────────
ax.set_xlim(0.5, 16.5)
ax.set_ylim(0, 0.11)

ax.set_xlabel("Epoch")
ax.set_ylabel("GEO-BLEU")

ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

# ── Legend ────────────────────────────────────────────────────────────────────
ax.legend(loc="lower right")

plt.title("GEO-BLEU per Epoch — Original vs Improved")
plt.tight_layout()
plt.show()