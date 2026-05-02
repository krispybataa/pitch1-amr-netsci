#!/usr/bin/env python3
"""
Step 4 — Statistical comparison + Visualizations.

4a: Mann-Whitney U tests (LOW vs HIGH temp) for centrality metrics + resistance rate.
    Output: data/processed/mannwhitney_results.csv

4b: 2x2 centrality boxplots with strip overlay + p-value annotations.
    Output: outputs/figures/centrality_boxplots.png

4c: NetworkX spring-layout graph coloured by temp group, sized by betweenness.
    Output: outputs/figures/amr_network_map.png

4d: Scatter — mean TXx vs mean resistance_rate, coloured by temp group + regression.
    Output: outputs/figures/temp_vs_resistance.png
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — write files, no display

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu, linregress

warnings.filterwarnings("ignore")

PROC_DIR  = Path("data/processed")
FIG_DIR   = Path("outputs/figures")
NET_DIR   = Path("outputs/networks")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Colour palette consistent across all plots
PALETTE = {"LOW": "#4878CF", "HIGH": "#D65F5F"}   # blue / red

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def pval_label(p: float) -> str:
    """Return a human-readable p-value string with significance stars."""
    if p < 0.001:
        return f"p={p:.2e} ***"
    if p < 0.01:
        return f"p={p:.4f} **"
    if p < 0.05:
        return f"p={p:.4f} *"
    return f"p={p:.4f} ns"


# --------------------------------------------------------------------------
# Step 4a — Mann-Whitney U tests
# --------------------------------------------------------------------------


def run_mannwhitney(
    scores: pd.DataFrame,
    master: pd.DataFrame,
) -> pd.DataFrame:
    """Return a DataFrame of MWU results for centrality metrics + resistance_rate."""

    # Per-country mean resistance rate (averaged across all years and combos)
    country_res = (
        master
        .groupby("iso3")["resistance_rate"]
        .mean()
        .reset_index()
        .rename(columns={"resistance_rate": "resistance_rate"})
    )
    scores_ext = scores.merge(country_res, on="iso3", how="left")

    metrics = ["degree", "betweenness", "closeness", "eigenvector", "resistance_rate"]
    rows = []

    for metric in metrics:
        low_vals  = scores_ext.loc[scores_ext["temp_group"] == "LOW",  metric].dropna().values
        high_vals = scores_ext.loc[scores_ext["temp_group"] == "HIGH", metric].dropna().values

        U, p = mannwhitneyu(low_vals, high_vals, alternative="two-sided")
        rows.append({
            "metric":       metric,
            "U_statistic":  U,
            "p_value":      p,
            "LOW_median":   float(np.median(low_vals)),
            "HIGH_median":  float(np.median(high_vals)),
            "LOW_n":        len(low_vals),
            "HIGH_n":       len(high_vals),
            "significant":  p < 0.05,
        })

    results = pd.DataFrame(rows)

    out = PROC_DIR / "mannwhitney_results.csv"
    results.to_csv(out, index=False)
    print(f"  Saved -> {out}")
    return results


def print_mwu_table(results: pd.DataFrame) -> None:
    print()
    print(f"  {'Metric':<20s}  {'U':>8s}  {'p-value':>10s}  {'LOW med':>9s}  {'HIGH med':>9s}  {'Sig?'}")
    print("  " + "-" * 74)
    for _, row in results.iterrows():
        sig = "YES *" if row["significant"] else "no"
        print(
            f"  {row['metric']:<20s}  {row['U_statistic']:>8.1f}  "
            f"{row['p_value']:>10.4f}  {row['LOW_median']:>9.4f}  "
            f"{row['HIGH_median']:>9.4f}  {sig}"
        )


# --------------------------------------------------------------------------
# Step 4b — Centrality boxplots
# --------------------------------------------------------------------------


def plot_centrality_boxplots(
    scores: pd.DataFrame,
    results: pd.DataFrame,
) -> None:
    metrics = ["degree", "betweenness", "closeness", "eigenvector"]
    titles  = [
        "Degree Centrality",
        "Betweenness Centrality",
        "Closeness Centrality",
        "Eigenvector Centrality",
    ]

    mwu_lookup = results.set_index("metric")["p_value"].to_dict()

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for ax, metric, title in zip(axes, metrics, titles):
        order = ["LOW", "HIGH"]

        # Box + whisker
        sns.boxplot(
            data=scores, x="temp_group", y=metric,
            order=order, palette=PALETTE,
            width=0.45, linewidth=1.4,
            flierprops=dict(marker="", linestyle="none"),   # hide default fliers
            ax=ax,
        )
        # Strip overlay (individual country dots)
        sns.stripplot(
            data=scores, x="temp_group", y=metric,
            order=order, palette=PALETTE,
            size=5, jitter=0.12, alpha=0.80,
            linewidth=0.5, edgecolor="white",
            ax=ax,
        )

        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel("Temperature group", fontsize=10)
        ax.set_ylabel(metric.capitalize(), fontsize=10)
        ax.tick_params(labelsize=9)

        # P-value annotation spanning the two groups
        p = mwu_lookup.get(metric, 1.0)
        y_max = scores[metric].max()
        y_range = scores[metric].max() - scores[metric].min()
        ann_y = y_max + 0.07 * y_range

        ax.annotate(
            pval_label(p),
            xy=(0.5, 0.97), xycoords="axes fraction",
            ha="center", va="top", fontsize=8.5,
            color="#333333",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", lw=0.8),
        )

    # Legend
    patches = [mpatches.Patch(color=PALETTE[g], label=f"{g} temp") for g in ["LOW", "HIGH"]]
    fig.legend(handles=patches, loc="lower center", ncol=2, fontsize=10,
               frameon=True, bbox_to_anchor=(0.5, -0.02))

    plt.suptitle(
        "AMR Network Centrality by Temperature Group\n(Mann-Whitney U test)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    out = FIG_DIR / "centrality_boxplots.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


# --------------------------------------------------------------------------
# Step 4c — Network map
# --------------------------------------------------------------------------


def plot_network(scores: pd.DataFrame) -> None:
    G = nx.read_gexf(str(NET_DIR / "amr_similarity_network.gexf"))

    # Node attribute lookup from centrality_scores (authoritative values)
    cent = scores.set_index("iso3")

    node_list  = list(G.nodes())
    node_color = [PALETTE[G.nodes[n]["temp_group"]] for n in node_list]

    # Betweenness: scale to node area (400–2800 px^2 range)
    bw_vals = np.array([cent.loc[n, "betweenness"] for n in node_list])
    bw_min, bw_max = bw_vals.min(), bw_vals.max()
    if bw_max > bw_min:
        node_size = 400 + 2400 * (bw_vals - bw_min) / (bw_max - bw_min)
    else:
        node_size = np.full(len(node_list), 800)

    # Edge widths: cosine similarity in (0.90, 1.00] → (0.3, 2.5]
    edge_list   = list(G.edges(data=True))
    edge_weights = np.array([d["weight"] for _, _, d in edge_list])
    edge_width   = 0.3 + 2.2 * (edge_weights - SIM_MIN) / (1.0 - SIM_MIN)

    # Layout: spring with fixed seed; increase k to reduce overlap
    pos = nx.spring_layout(G, seed=42, k=2.5 / np.sqrt(G.number_of_nodes()), iterations=120)

    fig, ax = plt.subplots(figsize=(13, 10))
    ax.set_facecolor("#f7f7f7")
    fig.patch.set_facecolor("#f7f7f7")

    # Edges
    nx.draw_networkx_edges(
        G, pos,
        edgelist=[(u, v) for u, v, _ in edge_list],
        width=edge_width,
        alpha=0.35,
        edge_color="#888888",
        ax=ax,
    )

    # Nodes
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=node_list,
        node_color=node_color,
        node_size=node_size,
        alpha=0.92,
        linewidths=1.2,
        edgecolors="white",
        ax=ax,
    )

    # Labels
    nx.draw_networkx_labels(
        G, pos,
        font_size=7.5, font_weight="bold", font_color="white",
        ax=ax,
    )

    # Legend: colour groups
    grp_patches = [
        mpatches.Patch(color=PALETTE["LOW"],  label="LOW temp group (n=13)"),
        mpatches.Patch(color=PALETTE["HIGH"], label="HIGH temp group (n=14)"),
    ]
    # Size legend: represent min, mid, max betweenness
    size_levels = [("Min bw", bw_min), ("Mid bw", (bw_min + bw_max) / 2), ("Max bw", bw_max)]
    size_patches = [
        plt.scatter([], [], s=400 + 2400 * (bw - bw_min) / max(bw_max - bw_min, 1e-9),
                    c="#888888", alpha=0.7, label=f"{label} ({bw:.3f})")
        for label, bw in size_levels
    ]

    leg1 = ax.legend(handles=grp_patches, loc="upper left",
                     fontsize=9, framealpha=0.9, title="Temperature group", title_fontsize=9)
    ax.add_artist(leg1)
    ax.legend(handles=size_patches, loc="upper right",
              fontsize=8, framealpha=0.9, title="Node size = betweenness", title_fontsize=9,
              scatterpoints=1)

    ax.set_title(
        "EU/EEA AMR Similarity Network\n"
        f"Node size: betweenness centrality  |  Edge width: cosine similarity  |  Threshold > {SIM_MIN}",
        fontsize=12, fontweight="bold", pad=12,
    )
    ax.axis("off")
    plt.tight_layout()

    out = FIG_DIR / "amr_network_map.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


# --------------------------------------------------------------------------
# Step 4d — Temperature vs resistance scatter
# --------------------------------------------------------------------------


def plot_temp_vs_resistance(
    master: pd.DataFrame,
    scores: pd.DataFrame,
    results: pd.DataFrame,
) -> None:
    # Country-level aggregates
    country_stats = (
        master
        .groupby("iso3")
        .agg(mean_TXx=("TXx", "mean"), mean_resistance=("resistance_rate", "mean"))
        .reset_index()
        .merge(scores[["iso3", "country_name", "temp_group"]], on="iso3")
    )

    # Linear regression
    x = country_stats["mean_TXx"].values
    y = country_stats["mean_resistance"].values
    slope, intercept, r, p_lr, _ = linregress(x, y)
    r2 = r ** 2
    x_line = np.linspace(x.min() - 0.5, x.max() + 0.5, 200)
    y_line = slope * x_line + intercept

    fig, ax = plt.subplots(figsize=(10, 7))

    # Scatter points
    for grp, grp_df in country_stats.groupby("temp_group"):
        ax.scatter(
            grp_df["mean_TXx"], grp_df["mean_resistance"],
            color=PALETTE[grp], s=80, zorder=4, alpha=0.90,
            label=f"{grp} temp (n={len(grp_df)})",
            edgecolors="white", linewidths=0.7,
        )

    # Country labels — offset slightly to reduce overlap
    for _, row in country_stats.iterrows():
        ax.annotate(
            row["iso3"],
            xy=(row["mean_TXx"], row["mean_resistance"]),
            xytext=(4, 4), textcoords="offset points",
            fontsize=7.5, color="#333333", fontweight="bold",
        )

    # Regression line
    ax.plot(x_line, y_line, color="#555555", lw=1.8, ls="--", zorder=3,
            label=f"OLS fit (R²={r2:.3f}, p={p_lr:.4f})")

    # Shaded 95% CI band via seaborn (light)
    sns.regplot(
        data=country_stats, x="mean_TXx", y="mean_resistance",
        scatter=False, ci=95,
        line_kws=dict(alpha=0),            # suppress seaborn's line; draw ours above
        ax=ax, color="#555555",
    )

    ax.set_xlabel("Mean Annual TXx (°C)  [Hottest Day Temperature]", fontsize=11)
    ax.set_ylabel("Mean AMR Resistance Rate (%)", fontsize=11)
    ax.set_title(
        "Country Mean Temperature vs AMR Resistance Rate\n"
        f"R²={r2:.3f}, slope={slope:.2f}%/°C, p={p_lr:.4f}",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9, framealpha=0.9)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3, lw=0.6)

    # Resistance MWU p-value annotation
    mwu_res_p = results.loc[results["metric"] == "resistance_rate", "p_value"].values[0]
    ax.text(
        0.02, 0.97,
        f"MWU LOW vs HIGH: {pval_label(mwu_res_p)}",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cccccc", lw=0.8),
    )

    plt.tight_layout()

    out = FIG_DIR / "temp_vs_resistance.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

SIM_MIN = 0.96   # minimum cosine similarity kept as edge (matches Step 3)


def main() -> None:
    master = pd.read_csv(PROC_DIR / "master_dataset.csv")
    scores = pd.read_csv(PROC_DIR / "centrality_scores.csv")

    print("=== Step 4a: Mann-Whitney U tests ===")
    results = run_mannwhitney(scores, master)
    print_mwu_table(results)

    print("\n=== Step 4b: Centrality boxplots ===")
    plot_centrality_boxplots(scores, results)

    print("\n=== Step 4c: Network map ===")
    plot_network(scores)

    print("\n=== Step 4d: Temperature vs resistance scatter ===")
    plot_temp_vs_resistance(master, scores, results)

    # Final significance summary
    sig = results[results["significant"]]
    not_sig = results[~results["significant"]]
    print()
    print("=== Significance summary ===")
    print(f"  Significant (p < 0.05): {sig['metric'].tolist()}")
    print(f"  Not significant:        {not_sig['metric'].tolist()}")


if __name__ == "__main__":
    main()
