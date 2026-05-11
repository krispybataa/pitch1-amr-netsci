#!/usr/bin/env python3
"""
Step 9 - Centrality confounder correlation analysis.
Examines which countries deviate from the OLS temp-resistance line
and whether centrality metrics explain those deviations.
Outputs:
  outputs/figures/residual_centrality_correlation.png
  outputs/figures/residual_outlier_table.png
  outputs/processed/regression_residuals.csv
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import linregress, spearmanr

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROC_DIR  = Path("data/processed")
OUT_FIGS  = Path("outputs/figures")
OUT_PROC  = Path("outputs/processed")
OUT_PROC.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_csv(path, required_cols):
    df = pd.read_csv(path)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"ERROR: {path} is missing columns: {missing}")
        sys.exit(1)
    return df

centrality = load_csv(PROC_DIR / "centrality_scores.csv",
                      ["iso3", "country_name", "temp_group", "mean_TXx",
                       "degree", "betweenness", "closeness", "eigenvector"])
resistance = load_csv(PROC_DIR / "resistance_matrix.csv", ["iso3"])

# ---------------------------------------------------------------------------
# Compute mean resistance rate per country
# ---------------------------------------------------------------------------

combo_cols = [c for c in resistance.columns if "|" in c]
resistance["mean_resistance"] = resistance[combo_cols].mean(axis=1)

# Merge into centrality frame
df = centrality.merge(
    resistance[["iso3", "mean_resistance"]], on="iso3", how="left"
)

missing_res = df["mean_resistance"].isna().sum()
if missing_res > 0:
    print(f"WARNING: {missing_res} countries have no resistance data after merge.")

df = df.dropna(subset=["mean_resistance", "mean_TXx"])

# ---------------------------------------------------------------------------
# OLS regression: mean_resistance ~ mean_TXx
# ---------------------------------------------------------------------------

x = df["mean_TXx"].values
y = df["mean_resistance"].values

slope, intercept, r, p_lr, se = linregress(x, y)
r2 = r ** 2

print(f"OLS: R2={r2:.4f}  slope={slope:.4f}  intercept={intercept:.4f}  p={p_lr:.4f}")

df["predicted_resistance"] = slope * df["mean_TXx"] + intercept
df["residual"] = df["mean_resistance"] - df["predicted_resistance"]

# ---------------------------------------------------------------------------
# 2x2 scatter panel: residuals vs each centrality metric
# ---------------------------------------------------------------------------

plt.style.use("seaborn-v0_8-whitegrid")
TEMP_COLORS = {"LOW": "#4878CF", "HIGH": "#D65F5F"}

metrics = [
    ("degree",      "Degree Centrality"),
    ("betweenness", "Betweenness Centrality"),
    ("closeness",   "Closeness Centrality"),
    ("eigenvector", "Eigenvector Centrality"),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
axes = axes.flatten()
fig.patch.set_facecolor("white")

for ax, (metric, title) in zip(axes, metrics):
    ax.set_facecolor("white")

    for grp, gdf in df.groupby("temp_group"):
        ax.scatter(
            gdf[metric], gdf["residual"],
            color=TEMP_COLORS[grp], s=70,
            alpha=0.85, edgecolors="white", linewidths=0.6,
            zorder=4, label=f"{grp} temp",
        )

    # ISO3 labels
    for _, row in df.iterrows():
        ax.annotate(
            row["iso3"],
            xy=(row[metric], row["residual"]),
            xytext=(4, 3), textcoords="offset points",
            fontsize=6.8, color="#333333",
        )

    # Horizontal reference line
    ax.axhline(0, color="#888888", lw=1.2, ls="--", alpha=0.7)

    # Spearman correlation
    rho, p_sp = spearmanr(df[metric], df["residual"])
    p_str = f"p={p_sp:.2e}" if p_sp < 0.001 else f"p={p_sp:.4f}"
    sig_star = ""
    if p_sp < 0.001:
        sig_star = " ***"
    elif p_sp < 0.01:
        sig_star = " **"
    elif p_sp < 0.05:
        sig_star = " *"
    ax.text(
        0.97, 0.97,
        f"Spearman r={rho:.3f}\n{p_str}{sig_star}",
        transform=ax.transAxes, fontsize=8.5,
        va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#BBBBBB", lw=0.8),
    )

    ax.set_xlabel(title, fontsize=10)
    ax.set_ylabel("Residual (actual - predicted resistance %)", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.tick_params(labelsize=9)

# Shared legend
patches = [mpatches.Patch(color=TEMP_COLORS[g], label=f"{g} temp")
           for g in ["LOW", "HIGH"]]
fig.legend(handles=patches, loc="lower center", ncol=2, fontsize=10,
           frameon=True, bbox_to_anchor=(0.5, -0.01))

fig.suptitle(
    f"OLS Residuals vs Centrality Metrics\n"
    f"(R²={r2:.3f}, slope={slope:.2f}%/°C, p={p_lr:.4f})",
    fontsize=13, fontweight="bold", y=1.01,
)
plt.tight_layout()

out_scatter = OUT_FIGS / "residual_centrality_correlation.png"
plt.savefig(out_scatter, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {out_scatter}")

# ---------------------------------------------------------------------------
# Outlier table figure
# ---------------------------------------------------------------------------

top5    = df.nlargest(5, "residual")
bottom5 = df.nsmallest(5, "residual")
outlier_df = pd.concat([top5, bottom5]).reset_index(drop=True)

table_cols = ["iso3", "country_name", "temp_group", "mean_TXx",
              "mean_resistance", "predicted_resistance", "residual",
              "eigenvector"]
table_data = outlier_df[table_cols].copy()

col_labels = ["ISO3", "Country", "Group", "TXx (C)",
              "Obs Res (%)", "Pred Res (%)", "Residual", "Eigenvector"]

cell_text = []
for _, row in table_data.iterrows():
    cell_text.append([
        row["iso3"],
        row["country_name"],
        row["temp_group"],
        f"{row['mean_TXx']:.2f}",
        f"{row['mean_resistance']:.2f}",
        f"{row['predicted_resistance']:.2f}",
        f"{row['residual']:+.2f}",
        f"{row['eigenvector']:.4f}",
    ])

fig_t, ax_t = plt.subplots(figsize=(14, 6))
fig_t.patch.set_facecolor("white")
ax_t.axis("off")

# Alternating row colors (top5 positive = light red, bottom5 negative = light blue)
row_colors = []
for i, (_, row) in enumerate(outlier_df.iterrows()):
    if row["residual"] >= 0:
        row_colors.append(["#FDECEA"] * len(col_labels))
    else:
        row_colors.append(["#EAF1FB"] * len(col_labels))

tbl = ax_t.table(
    cellText=cell_text,
    colLabels=col_labels,
    cellLoc="center",
    loc="center",
    cellColours=row_colors,
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.6)

for j in range(len(col_labels)):
    tbl[0, j].set_facecolor("#2C3E50")
    tbl[0, j].set_text_props(color="white", fontweight="bold")

ax_t.set_title(
    "Regression Residual Outliers\n"
    "Top 5 (more resistant than expected) & Bottom 5 (less resistant than expected)",
    fontsize=12, fontweight="bold", pad=20,
)
plt.tight_layout()

out_table = OUT_FIGS / "residual_outlier_table.png"
plt.savefig(out_table, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {out_table}")

# ---------------------------------------------------------------------------
# Residuals CSV
# ---------------------------------------------------------------------------

out_cols = ["iso3", "country_name", "temp_group", "mean_TXx",
            "mean_resistance", "predicted_resistance", "residual",
            "degree", "betweenness", "closeness", "eigenvector"]
out_df = df[out_cols].sort_values("residual", ascending=False).reset_index(drop=True)

out_csv = OUT_PROC / "regression_residuals.csv"
out_df.to_csv(out_csv, index=False)
print(f"Saved: {out_csv}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

files = [out_scatter, out_table, out_csv]
print("\n--- Output summary ---")
for f in files:
    size_kb = Path(f).stat().st_size // 1024
    print(f"  {f}  ({size_kb} KB)")
