#!/usr/bin/env python3
"""
Step 7 - Community overlay comparison: two-panel network figure.
Left: nodes colored by Louvain community. Right: nodes colored by temp group.
Outputs:
  outputs/figures/community_overlay_comparison.png
  outputs/processed/community_composition.csv
"""

import sys
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

resistance = load_csv(PROC_DIR / "resistance_matrix.csv",
                      ["iso3", "country_name"])
centrality = load_csv(PROC_DIR / "centrality_scores.csv",
                      ["iso3", "temp_group", "mean_TXx", "degree",
                       "betweenness", "closeness", "eigenvector"])
community  = load_csv(PROC_DIR / "community_assignments.csv",
                      ["iso3", "community", "temp_group"])

# ---------------------------------------------------------------------------
# Rebuild cosine similarity graph
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

print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges "
      f"(threshold > {SIM_THRESHOLD})")

# ---------------------------------------------------------------------------
# Merge node attributes
# ---------------------------------------------------------------------------

node_data = (
    community[["iso3", "community", "temp_group"]]
    .merge(centrality[["iso3", "mean_TXx", "degree", "betweenness",
                        "closeness", "eigenvector"]], on="iso3", how="left")
)

for _, row in node_data.iterrows():
    iso3 = row["iso3"]
    if iso3 in G.nodes:
        for col in ["community", "temp_group", "mean_TXx", "degree",
                    "betweenness", "closeness", "eigenvector"]:
            G.nodes[iso3][col] = row[col]

# ---------------------------------------------------------------------------
# Layout (fixed seed for reproducibility)
# ---------------------------------------------------------------------------

pos = nx.spring_layout(G, seed=42, k=2.5, iterations=150)
node_list = list(G.nodes())

# ---------------------------------------------------------------------------
# Node sizes (betweenness)
# ---------------------------------------------------------------------------

bw_vals = np.array([G.nodes[n]["betweenness"] for n in node_list])
bw_min, bw_max = bw_vals.min(), bw_vals.max()
if bw_max > bw_min:
    node_sizes = 400 + 2400 * (bw_vals - bw_min) / (bw_max - bw_min)
else:
    node_sizes = np.full(len(node_list), 800)

# ---------------------------------------------------------------------------
# Edge widths (cosine similarity)
# ---------------------------------------------------------------------------

edge_list    = list(G.edges(data=True))
edge_weights = np.array([d["weight"] for _, _, d in edge_list])
w_min, w_max = edge_weights.min(), edge_weights.max()
if w_max > w_min:
    edge_widths = 0.5 + 2.0 * (edge_weights - w_min) / (w_max - w_min)
else:
    edge_widths = np.full(len(edge_weights), 1.25)

edge_nodes = [(u, v) for u, v, _ in edge_list]

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

COMM_COLORS = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c", 3: "#d62728"}
TEMP_COLORS = {"LOW": "#4878CF", "HIGH": "#D65F5F"}

comm_node_colors = [COMM_COLORS[G.nodes[n]["community"]] for n in node_list]
temp_node_colors = [TEMP_COLORS[G.nodes[n]["temp_group"]] for n in node_list]

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(20, 9))
fig.patch.set_facecolor("white")

draw_kwargs = dict(
    edgelist=edge_nodes,
    width=edge_widths,
    edge_color="#AAAAAA",
    alpha=0.4,
)

label_kwargs = dict(
    labels={n: n for n in node_list},
    font_size=7.5,
    font_weight="bold",
    font_color="white",
)

for ax, node_colors, title in [
    (axes[0], comm_node_colors, "AMR Network — Louvain Communities"),
    (axes[1], temp_node_colors, "AMR Network — Temperature Groups"),
]:
    ax.set_facecolor("white")
    nx.draw_networkx_edges(G, pos, ax=ax, **draw_kwargs)
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=node_list,
        node_color=node_colors,
        node_size=node_sizes,
        linewidths=0.8,
        edgecolors="white",
        ax=ax,
    )
    nx.draw_networkx_labels(G, pos, ax=ax, **label_kwargs)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.axis("off")

# Community legend
comm_handles = [
    mpatches.Patch(facecolor=COMM_COLORS[c],
                   label=f"Community {c}")
    for c in sorted(COMM_COLORS)
]
axes[0].legend(handles=comm_handles, loc="lower left", fontsize=9,
               framealpha=0.9, title="Community", title_fontsize=9)

# Temp legend
temp_handles = [
    mpatches.Patch(facecolor=TEMP_COLORS[g], label=f"{g} temp")
    for g in ["LOW", "HIGH"]
]
axes[1].legend(handles=temp_handles, loc="lower left", fontsize=9,
               framealpha=0.9, title="Temperature group", title_fontsize=9)

# Size legend (shared)
size_ref = [("Min betweenness", bw_min), ("Max betweenness", bw_max)]
size_handles = [
    plt.scatter([], [],
                s=400 + 2400 * (bw - bw_min) / max(bw_max - bw_min, 1e-9),
                c="#888888", alpha=0.7,
                label=f"{lbl} ({bw:.3f})")
    for lbl, bw in size_ref
]
axes[1].legend(
    handles=temp_handles + size_handles,
    loc="lower left", fontsize=9, framealpha=0.9,
    title="Temp group / Node size = betweenness", title_fontsize=9,
)

fig.suptitle(
    "AMR Cosine-Similarity Network (threshold > 0.96)\n"
    "Node size: betweenness centrality",
    fontsize=14, fontweight="bold", y=1.01,
)
plt.tight_layout(pad=1.5)

out_fig = OUT_FIGS / "community_overlay_comparison.png"
plt.savefig(out_fig, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {out_fig}")

# ---------------------------------------------------------------------------
# Community composition CSV
# ---------------------------------------------------------------------------

comp = node_data[["community", "iso3", "temp_group", "mean_TXx", "betweenness"]].copy()
comp = comp.sort_values(["community", "mean_TXx"]).reset_index(drop=True)

out_csv = OUT_PROC / "community_composition.csv"
comp.to_csv(out_csv, index=False)
print(f"Saved: {out_csv}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

files = [out_fig, out_csv]
print("\n--- Output summary ---")
for f in files:
    size_kb = Path(f).stat().st_size // 1024
    print(f"  {f}  ({size_kb} KB)")
