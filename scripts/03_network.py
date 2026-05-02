#!/usr/bin/env python3
"""
Step 3 — Temperature quantile binning + AMR similarity network construction.

3a: Bin countries into LOW / HIGH temperature groups based on median mean-TXx.
    Output: data/processed/country_temp_groups.csv

3b: Build country × pathogen-antibiotic mean-resistance matrix.
    Output: data/processed/resistance_matrix.csv

3c: Cosine-similarity network (threshold 0.90), exported to GEXF for Gephi.
    Output: outputs/networks/amr_similarity_network.gexf

3d: Node-level centrality metrics.
    Output: data/processed/centrality_scores.csv
"""

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

PROC_DIR = Path("data/processed")
OUT_DIR  = Path("outputs/networks")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SIM_THRESHOLD = 0.90   # edges with cosine similarity <= this are dropped

# ---------------------------------------------------------------------------
# Step 3a — Temperature binning
# ---------------------------------------------------------------------------


def bin_temperature(master: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with iso3, country_name, mean_TXx, temp_group."""
    country_tx = (
        master
        .groupby(["iso3", "country_name"], as_index=False)["TXx"]
        .mean()
        .rename(columns={"TXx": "mean_TXx"})
        .sort_values("mean_TXx")
        .reset_index(drop=True)
    )

    median_tx = country_tx["mean_TXx"].median()
    country_tx["temp_group"] = np.where(
        country_tx["mean_TXx"] < median_tx, "LOW", "HIGH"
    )

    print(f"  Median mean TXx across countries: {median_tx:.3f} C")
    print()
    for grp in ["LOW", "HIGH"]:
        subset = country_tx[country_tx["temp_group"] == grp]
        print(f"  {grp} ({len(subset)} countries):")
        for _, row in subset.iterrows():
            print(f"    {row['iso3']}  {row['country_name']:<25s}  {row['mean_TXx']:.3f} C")
        print()

    out = PROC_DIR / "country_temp_groups.csv"
    country_tx.to_csv(out, index=False)
    print(f"  Saved -> {out}")
    return country_tx


# ---------------------------------------------------------------------------
# Step 3b — Resistance matrix
# ---------------------------------------------------------------------------


def build_resistance_matrix(master: pd.DataFrame) -> pd.DataFrame:
    """
    Country x pathogen-antibiotic mean-resistance matrix.
    Columns: iso3, country_name, then one column per combo.
    """
    # Mean resistance per country × combo (averaged across years)
    pivot = master.pivot_table(
        index=["iso3", "country_name"],
        columns=["pathogen", "antibiotic"],
        values="resistance_rate",
        aggfunc="mean",
    )

    # Flatten MultiIndex columns to "Pathogen|Antibiotic"
    pivot.columns = [f"{p}|{a}" for p, a in pivot.columns]
    pivot = pivot.reset_index()

    combo_cols = [c for c in pivot.columns if "|" in c]

    # Drop rows where more than 50% of combo values are NaN
    row_null_frac = pivot[combo_cols].isnull().mean(axis=1)
    n_before = len(pivot)
    pivot = pivot[row_null_frac <= 0.50].copy()
    n_dropped = n_before - len(pivot)
    if n_dropped:
        print(f"  Dropped {n_dropped} country rows with >50% NaN combos.")

    # Fill remaining NaNs with column (combo) means
    col_means = pivot[combo_cols].mean()
    pivot[combo_cols] = pivot[combo_cols].fillna(col_means)

    print(f"  Resistance matrix: {len(pivot)} countries x {len(combo_cols)} combos")
    print(f"  NaN count after fill: {pivot[combo_cols].isnull().sum().sum()}")

    out = PROC_DIR / "resistance_matrix.csv"
    pivot.to_csv(out, index=False)
    print(f"  Saved -> {out}")
    return pivot


# ---------------------------------------------------------------------------
# Step 3c — Cosine similarity network
# ---------------------------------------------------------------------------


def build_network(
    resistance_matrix: pd.DataFrame,
    temp_groups: pd.DataFrame,
    threshold: float = SIM_THRESHOLD,
) -> nx.Graph:
    """Build and return a weighted NetworkX graph from cosine similarity."""
    combo_cols = [c for c in resistance_matrix.columns if "|" in c]
    countries  = resistance_matrix["iso3"].tolist()
    names      = resistance_matrix["country_name"].tolist()
    X = resistance_matrix[combo_cols].values  # (n_countries, n_combos)

    sim_matrix = cosine_similarity(X)  # (n_countries, n_countries)

    # Map iso3 -> node attributes
    tg_lookup = temp_groups.set_index("iso3")

    G = nx.Graph()

    # Add nodes with attributes
    for i, iso3 in enumerate(countries):
        row = tg_lookup.loc[iso3]
        G.add_node(
            iso3,
            country_name=names[i],
            temp_group=row["temp_group"],
            mean_TXx=float(row["mean_TXx"]),
        )

    # Add edges above threshold
    n = len(countries)
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(sim_matrix[i, j])
            if sim > threshold:
                G.add_edge(countries[i], countries[j], weight=sim)

    # Summary
    density = nx.density(G)
    degrees = dict(G.degree())
    deg_values = list(degrees.values())
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()} (threshold > {threshold})")
    print(f"  Density: {density:.4f}")
    print(f"  Degree: min={min(deg_values)}  mean={np.mean(deg_values):.1f}  max={max(deg_values)}")
    print(f"  Connected: {nx.is_connected(G)}")
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        print(f"  Components: {len(components)}, sizes: {sorted([len(c) for c in components], reverse=True)}")

    # Export GEXF for Gephi
    gexf_path = OUT_DIR / "amr_similarity_network.gexf"
    nx.write_gexf(G, str(gexf_path))
    print(f"  Saved GEXF -> {gexf_path}")

    return G


# ---------------------------------------------------------------------------
# Step 3d — Centrality metrics
# ---------------------------------------------------------------------------


def compute_centrality(
    G: nx.Graph,
    temp_groups: pd.DataFrame,
) -> pd.DataFrame:
    """Compute four centrality metrics for every node and return a DataFrame."""
    degree_c      = nx.degree_centrality(G)
    betweenness_c = nx.betweenness_centrality(G, weight="weight", normalized=True)
    closeness_c   = nx.closeness_centrality(G, distance="weight")
    try:
        eigenvector_c = nx.eigenvector_centrality(G, weight="weight", max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        print("  WARNING: eigenvector centrality did not converge; using degree as proxy.")
        eigenvector_c = degree_c

    tg_lookup = temp_groups.set_index("iso3")

    rows = []
    for iso3 in G.nodes:
        node_data = G.nodes[iso3]
        rows.append({
            "iso3":          iso3,
            "country_name":  node_data["country_name"],
            "temp_group":    node_data["temp_group"],
            "mean_TXx":      node_data["mean_TXx"],
            "degree":        degree_c[iso3],
            "betweenness":   betweenness_c[iso3],
            "closeness":     closeness_c[iso3],
            "eigenvector":   eigenvector_c[iso3],
        })

    scores = (
        pd.DataFrame(rows)
        .sort_values("betweenness", ascending=False)
        .reset_index(drop=True)
    )

    out = PROC_DIR / "centrality_scores.csv"
    scores.to_csv(out, index=False)
    print(f"  Saved -> {out}")

    print()
    print("  Top 10 countries by betweenness centrality:")
    cols = ["iso3", "country_name", "temp_group", "betweenness", "degree", "eigenvector"]
    print(scores.head(10)[cols].to_string(index=False))

    return scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    master = pd.read_csv(PROC_DIR / "master_dataset.csv")

    print("=== Step 3a: Temperature binning ===")
    temp_groups = bin_temperature(master)

    print("\n=== Step 3b: Resistance matrix ===")
    resistance_matrix = build_resistance_matrix(master)

    print("\n=== Step 3c: Cosine similarity network ===")
    G = build_network(resistance_matrix, temp_groups)

    print("\n=== Step 3d: Centrality metrics ===")
    scores = compute_centrality(G, temp_groups)

    # Summary by temp group
    print()
    print("=== Centrality summary by temperature group ===")
    summary = scores.groupby("temp_group")[["degree", "betweenness", "closeness", "eigenvector"]].mean()
    print(summary.to_string())


if __name__ == "__main__":
    main()
