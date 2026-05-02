#!/usr/bin/env python3
"""
Step 5 — Publication-quality AMR network figure.
Output: outputs/figures/amr_network_final.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
from pathlib import Path

GEXF_PATH = Path("outputs/networks/amr_similarity_network.gexf")
OUT_PATH  = Path("outputs/figures/amr_network_final.png")

PALETTE = {"LOW": "#4A90D9", "HIGH": "#E74C3C"}

NODE_SIZE_MIN, NODE_SIZE_MAX = 800, 3000
EDGE_W_MIN,   EDGE_W_MAX    = 0.5, 2.5

# ---------------------------------------------------------------------------
# Load graph
# ---------------------------------------------------------------------------

G = nx.read_gexf(str(GEXF_PATH))

node_list = list(G.nodes())
edge_list  = list(G.edges(data=True))

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

pos = nx.spring_layout(G, seed=42, k=2.5, iterations=150)

# ---------------------------------------------------------------------------
# Node attributes
# ---------------------------------------------------------------------------

degree_vals = np.array([G.nodes[n]["degree"] for n in node_list])
d_min, d_max = degree_vals.min(), degree_vals.max()
node_sizes = NODE_SIZE_MIN + (NODE_SIZE_MAX - NODE_SIZE_MIN) * (
    (degree_vals - d_min) / (d_max - d_min)
)

node_colors = [PALETTE[G.nodes[n]["temp_group"]] for n in node_list]

# ---------------------------------------------------------------------------
# Edge attributes
# ---------------------------------------------------------------------------

edge_nodes  = [(u, v) for u, v, _ in edge_list]
raw_weights = np.array([d["weight"] for _, _, d in edge_list])
w_min, w_max = raw_weights.min(), raw_weights.max()
if w_max > w_min:
    edge_widths = EDGE_W_MIN + (EDGE_W_MAX - EDGE_W_MIN) * (
        (raw_weights - w_min) / (w_max - w_min)
    )
else:
    edge_widths = np.full(len(raw_weights), (EDGE_W_MIN + EDGE_W_MAX) / 2)

# ---------------------------------------------------------------------------
# Draw
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(14, 10))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# Edges
nx.draw_networkx_edges(
    G, pos,
    edgelist=edge_nodes,
    width=edge_widths,
    edge_color="#CCCCCC",
    alpha=0.85,
    ax=ax,
)

# Nodes
nx.draw_networkx_nodes(
    G, pos,
    nodelist=node_list,
    node_color=node_colors,
    node_size=node_sizes,
    linewidths=0.8,
    edgecolors="white",
    ax=ax,
)

# Labels
nx.draw_networkx_labels(
    G, pos,
    labels={n: n for n in node_list},
    font_size=8,
    font_weight="bold",
    font_color="white",
    ax=ax,
)

# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------

legend_handles = [
    mpatches.Patch(facecolor=PALETTE["LOW"],  edgecolor="white", label="LOW temperature"),
    mpatches.Patch(facecolor=PALETTE["HIGH"], edgecolor="white", label="HIGH temperature"),
]
ax.legend(
    handles=legend_handles,
    loc="lower left",
    fontsize=11,
    frameon=True,
    framealpha=0.9,
    edgecolor="#DDDDDD",
    title="Temperature group",
    title_fontsize=11,
)

ax.axis("off")
plt.tight_layout(pad=1.0)

plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()

size_kb = OUT_PATH.stat().st_size // 1024
print(f"Saved -> {OUT_PATH}  ({size_kb} KB)")
