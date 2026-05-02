#!/usr/bin/env python3
"""
Step 6 — Louvain community detection on the AMR similarity network.

Outputs:
  data/processed/community_assignments.csv
  outputs/networks/amr_similarity_network.gexf  (updated with community attr)
"""

import community as louvain        # python-louvain package
import networkx as nx
import pandas as pd
from pathlib import Path

GEXF_PATH = Path("outputs/networks/amr_similarity_network.gexf")
CSV_PATH  = Path("data/processed/community_assignments.csv")

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Load graph
# ---------------------------------------------------------------------------

G = nx.read_gexf(str(GEXF_PATH))
print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# ---------------------------------------------------------------------------
# Louvain community detection
# ---------------------------------------------------------------------------

partition = louvain.best_partition(G, weight="weight", random_state=RANDOM_STATE)
# partition: {node_id: community_id}

n_communities = len(set(partition.values()))
modularity    = louvain.modularity(partition, G, weight="weight")

print(f"\nLouvain results")
print(f"  Communities : {n_communities}")
print(f"  Modularity  : {modularity:.4f}")

# ---------------------------------------------------------------------------
# Build community assignment table
# ---------------------------------------------------------------------------

rows = []
for iso3, comm_id in partition.items():
    node = G.nodes[iso3]
    rows.append({
        "iso3":         iso3,
        "country_name": node["country_name"],
        "temp_group":   node["temp_group"],
        "mean_TXx":     node["mean_TXx"],
        "community":    comm_id,
    })

assignments = (
    pd.DataFrame(rows)
    .sort_values(["community", "mean_TXx"])
    .reset_index(drop=True)
)

# ---------------------------------------------------------------------------
# Print community breakdown
# ---------------------------------------------------------------------------

print()
for comm_id in sorted(assignments["community"].unique()):
    members = assignments[assignments["community"] == comm_id]
    n       = len(members)
    n_high  = (members["temp_group"] == "HIGH").sum()
    n_low   = (members["temp_group"] == "LOW").sum()
    pct_high = 100 * n_high / n
    pct_low  = 100 * n_low  / n

    country_list = "  ".join(
        f"{row['iso3']} ({row['temp_group']})"
        for _, row in members.iterrows()
    )

    print(f"Community {comm_id}  [{n} countries | HIGH={n_high} ({pct_high:.0f}%) | LOW={n_low} ({pct_low:.0f}%)]")
    print(f"  {country_list}")
    print()

# ---------------------------------------------------------------------------
# Write community back onto graph nodes, re-export GEXF
# ---------------------------------------------------------------------------

for iso3, comm_id in partition.items():
    G.nodes[iso3]["community"] = int(comm_id)

nx.write_gexf(G, str(GEXF_PATH))
print(f"Saved GEXF (with community attr) -> {GEXF_PATH}")

# ---------------------------------------------------------------------------
# Save CSV
# ---------------------------------------------------------------------------

assignments.to_csv(CSV_PATH, index=False)
print(f"Saved CSV                        -> {CSV_PATH}")
