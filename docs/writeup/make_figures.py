"""Column-width figures for the 2-page write-up.

Run from the repo root after `python -m src.experiments`:
    python docs/writeup/make_figures.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config as cfg
from src.data import load_city

OUT = Path(__file__).parent
INK, MUTED, GRID, BLUE, LIGHT, GREY = "#2C2C2B", "#7D7A75", "#E6E5E3", "#2783DE", "#A9CDF2", "#C9C7C3"
CITY = {"nyc": ("New York City", "#2783DE"), "paris": ("Paris", "#D5A23B"), "berlin": ("Berlin", "#46A171")}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.edgecolor": GRID, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.axisbelow": True, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "figure.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

# Figure 1: log-price distribution per city
fig, ax = plt.subplots(figsize=(3.45, 1.75))
bins = np.linspace(np.log(10), np.log(3000), 70)
legend_labels = []
for c, (name, col) in CITY.items():
    p = load_city(c)["price_usd"]
    legend_labels.append(f"{name.replace('New York City', 'NYC')} · median ${p.median():.0f}")
    ax.hist(np.log(p), bins=bins, density=True, histtype="step", lw=1.3, color=col,
            label=f"{name} (median ${p.median():.0f})")
ticks = [10, 30, 100, 300, 1000, 3000]
ax.set_xticks(np.log(ticks)); ax.set_xticklabels([f"${t:,}" for t in ticks])
ax.set_xlabel("Nightly price (USD, log scale)"); ax.set_yticks([]); ax.grid(axis="y", visible=False)
ax.spines["left"].set_visible(False)
from matplotlib.lines import Line2D
handles = [Line2D([], [], color=col, lw=1.6) for _, col in CITY.values()]
labels = legend_labels
ax.legend(handles, labels, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0),
          ncol=3, fontsize=6.8, handlelength=1.2, columnspacing=1.0, borderaxespad=0.1)
fig.savefig(OUT / "fig1_price.png", facecolor="white"); plt.close(fig)

# Figure 2: transfer to Berlin (Table 8)
t8 = pd.read_csv(cfg.TABLES_DIR / "T8.csv") if (cfg.TABLES_DIR / "T8.csv").exists() else \
    pd.read_csv(cfg.RESULTS_DIR / "all_results.csv").query("table == 'T8'")
def pick(model, train, feat):
    r = t8[(t8.model.str.startswith(model)) & (t8.train_data == train) & (t8.features == feat)]
    if r.empty:
        raise SystemExit(f"T8 row not found: {model} | {train} | {feat} - re-run src.experiments")
    r = r.iloc[0]
    lo, hi = r.get("test_r2_ci_low", np.nan), r.get("test_r2_ci_high", np.nan)
    return float(r.test_r2), float(lo), float(hi)
NONE_C, LEVEL_C, ROWS_C = "#C9DDF2", BLUE, "#8FB8E3"
rows = [  # (label, (r2, ci_lo, ci_hi), colour) - grouped by the Berlin information used
    ("No Berlin info · NN, paper features", pick("Neural", "NYC + Paris", "C+O (paper)"), NONE_C),
    ("No Berlin info · XGB, city-agnostic", pick("XGB", "NYC + Paris", "C+O city-agnostic"), NONE_C),
    ("Price level · XGB, paper features", pick("XGB", "NYC + Paris", "C+O (paper) + price level"), LEVEL_C),
    ("Price level · XGB, city-agnostic", pick("XGB", "NYC + Paris", "C+O agnostic + price level"), LEVEL_C),
    ("10% rows · XGB, NYC+Paris+Berlin", pick("XGB", "NYC + Paris + 10% Berlin", "C+O city-agnostic"), ROWS_C),
    ("10% rows + level · XGB", pick("XGB", "NYC + Paris + 10% Berlin", "C+O agnostic + price level"), ROWS_C),
    ("10% rows · XGB, Berlin sample only", pick("XGB", "10% Berlin sample only", "C+O"), ROWS_C),
    ("Full Berlin train · XGB (reference)", pick("XGB", "Berlin only (in-city reference)", "C+O"), GREY),
]
fig, ax = plt.subplots(figsize=(3.45, 2.2))
y = np.arange(len(rows))[::-1]
vals = np.array([r[1][0] for r in rows])
lo = np.array([r[1][1] for r in rows]); hi = np.array([r[1][2] for r in rows])
xerr = None if np.isnan(lo).any() else np.vstack([vals - lo, hi - vals])
ax.barh(y, np.clip(vals, 0, None), color=[r[2] for r in rows], height=0.62,
        xerr=xerr, error_kw=dict(lw=0.6, capsize=1.5, ecolor=MUTED))
for yi, v, h in zip(y, vals, hi if xerr is not None else vals):
    ax.text(max(h, 0) + 0.012, yi, f"{v:.3f}", va="center", fontsize=7, color=INK)
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
ax.set_xlim(0, 0.95); ax.set_xlabel("R² on Berlin test split (log price), 95% CI"); ax.grid(axis="y", visible=False)
ax.tick_params(axis="y", length=0)
fig.savefig(OUT / "fig2_transfer.png", facecolor="white"); plt.close(fig)
print("figures written to", OUT)
