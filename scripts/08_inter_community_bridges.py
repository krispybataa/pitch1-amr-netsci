#!/usr/bin/env python3
"""
Step 8 - Inter-community bridge analysis using participation coefficient.
Outputs:
  outputs/figures/bridge_participation.png
  outputs/processed/bridge_analysis.csv
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROC_DIR  = Path("data/processed")
OUT_FIGS  = Path("outputs/figures")
OUT_PROC  = Path("outputs/processed")
OUT_PROC.mkdir(parents=True, exist_ok=True)

SIM_THRESHOLD = 0.96

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

resistance = load_csv(PROC_DIR / "resistance_matrix.csv", ["iso3"])
centrality = load_csv(PROC_DIR / "centrality_scores.csv",
                      ["iso3", "temp_group", "degree", "betweenness",
                       "closeness", "eigenvector"])
community  = load_csv(PROC_DIR / "community_assignments.csv",
                      ["iso3", "community", "temp_group"])

# ---------------------------------------------------------------------------
# Rebuild graph
# ---------------------------------------------------------------------------

combo_cols = [c for c in resistance.columns if "|" in c]
countries  = resistance["iso3"].tolist()
X          = resistance[combo_cols].values
sim_matrix = cosine_similarity(X)

G = nx.Graph()
for iso3 in countries:
    G.add_node(iso3)

n = len(countries)
for i in range(n):
    for j in range(i + 1, n):
        sim = float(sim_matrix[i, j])
        if sim > SIM_THRESHOLD:
            G.add_edge(countries[i], countries[j], weight=sim)

print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Assign community as node attribute
comm_lookup = community.set_index("iso3")["community"].to_dict()
temp_lookup = community.set_index("iso3")["temp_group"].to_dict()
for iso3 in G.nodes:
    G.nodes[iso3]["community"] = comm_lookup.get(iso3, -1)
    G.nodes[iso3]["temp_group"] = temp_lookup.get(iso3, "UNKNOWN")

# ---------------------------------------------------------------------------
# Participation coefficient (Guimera & Amaral 2005)
# ---------------------------------------------------------------------------

all_communities = sorted(set(comm_lookup.values()))

rows = []
for iso3 in G.nodes:
    ki = G.degree(iso3)
    if ki == 0:
        pc = 0.0
        intra = 0
        inter = 0
    else:
        # Count edges to each community
        comm_i = G.nodes[iso3]["community"]
        comm_counts = {c: 0 for c in all_communities}
        for nbr in G.neighbors(iso3):
            comm_nbr = G.nodes[nbr]["community"]
            comm_counts[comm_nbr] += 1

        intra = comm_counts[comm_i]
        inter = ki - intra

        pc = 1.0 - sum((k_is / ki) ** 2 for k_is in comm_counts.values())

    rows.append({
        "iso3":              iso3,
        "temp_group":        G.nodes[iso3]["temp_group"],
        "community":         G.nodes[iso3]["community"],
        "intra_degree":      intra,
        "inter_degree":      inter,
        "participation_coeff": round(pc, 6),
    })

bridge_df = pd.DataFrame(rows)

# Merge centrality metrics
bridge_df = bridge_df.merge(
    centrality[["iso3", "degree", "betweenness", "closeness", "eigenvector"]],
    on="iso3", how="left",
)

# ---------------------------------------------------------------------------
# Node role classification
# ---------------------------------------------------------------------------

med_degree = bridge_df["degree"].median()
med_pc     = bridge_df["participation_coeff"].median()

def classify(row):
    high_deg = row["degree"] >= med_degree
    high_pc  = row["participation_coeff"] >= med_pc
    if high_deg and high_pc:
        return "Connector hub"
    if high_deg and not high_pc:
        return "Provincial hub"
    if not high_deg and high_pc:
        return "Kinless"
    return "Peripheral"

bridge_df["node_role"] = bridge_df.apply(classify, axis=1)

print(f"\nMedian degree: {med_degree:.4f}  |  Median participation: {med_pc:.4f}")
print("\nNode role distribution:")
print(bridge_df["node_role"].value_counts().to_string())

# ---------------------------------------------------------------------------
# Scatter plot
# ---------------------------------------------------------------------------

TEMP_COLORS = {"LOW": "#4878CF", "HIGH": "#D65F5F"}

plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(12, 8))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

for grp, gdf in bridge_df.groupby("temp_group"):
    ax.scatter(
        gdf["betweenness"],
        gdf["participation_coeff"],
        c=TEMP_COLORS[grp],
        s=gdf["degree"] * 800,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.8,
        zorder=4,
        label=f"{grp} temp",
    )

# Labels
for _, row in bridge_df.iterrows():
    ax.annotate(
        row["iso3"],
        xy=(row["betweenness"], row["participation_coeff"]),
        xytext=(5, 4),
        textcoords="offset points",
        fontsize=7.5,
        fontweight="bold",
        color="#222222",
    )

# Median reference lines
ax.axhline(med_pc,     color="#888888", lw=1.2, ls="--", alpha=0.7,
           label=f"Median P ({med_pc:.3f})")
ax.axvline(med_degree, color="#555555", lw=1.2, ls=":",  alpha=0.7,
           label=f"Median degree ({med_degree:.3f})")

# Quadrant labels
y_top  = bridge_df["participation_coeff"].max()
x_max  = bridge_df["betweenness"].max()
ax.text(x_max * 0.02, y_top * 0.97, "Kinless",
        fontsize=9, color="#666666", va="top")
ax.text(x_max * 0.97, y_top * 0.97, "Connector hub",
        fontsize=9, color="#666666", va="top", ha="right")
ax.text(x_max * 0.02, med_pc * 0.05, "Peripheral",
        fontsize=9, color="#666666", va="bottom")
ax.text(x_max * 0.97, med_pc * 0.05, "Provincial hub",
        fontsize=9, color="#666666", va="bottom", ha="right")

# Top 3 connectors annotation box
top3 = bridge_df.nlargest(3, "participation_coeff")[["iso3", "participation_coeff"]]
ann_text = "Top 3 connector nodes:\n" + "\n".join(
    f"  {r['iso3']}  P={r['participation_coeff']:.3f}"
    for _, r in top3.iterrows()
)
ax.text(
    0.98, 0.02, ann_text,
    transform=ax.transAxes,
    fontsize=8.5,
    va="bottom", ha="right",
    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#BBBBBB", lw=0.9),
)

ax.set_xlabel("Betweenness Centrality", fontsize=12)
ax.set_ylabel("Participation Coefficient (P)", fontsize=12)
ax.set_title(
    "Inter-community Bridge Analysis\n"
    "Participation coefficient vs betweenness centrality",
    fontsize=13, fontweight="bold",
)
ax.legend(fontsize=9, framealpha=0.9, loc="upper left")
plt.tight_layout()

out_fig = OUT_FIGS / "bridge_participation.png"
plt.savefig(out_fig, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\nSaved: {out_fig}")

# ---------------------------------------------------------------------------
# Bridge analysis CSV
# ---------------------------------------------------------------------------

out_cols = ["iso3", "temp_group", "community", "degree", "betweenness",
            "participation_coeff", "node_role"]
out_df = (
    bridge_df[out_cols]
    .sort_values("participation_coeff", ascending=False)
    .reset_index(drop=True)
)
out_csv = OUT_PROC / "bridge_analysis.csv"
out_df.to_csv(out_csv, index=False)
print(f"Saved: {out_csv}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

files = [out_fig, out_csv]
print("\n--- Output summary ---")
for f in files:
    size_kb = Path(f).stat().st_size // 1024
    print(f"  {f}  ({size_kb} KB)")
