# Pitch 1 — AMR Network Topology & Climate Change

**Research Question:** Do higher temperatures increase the connectivity of antimicrobial resistance (AMR) profiles across EU/EEA countries? Which countries act as inter-community bridges within the resistance similarity network?

## Team

| Name | Role |
|---|---|
| Clark Rodriguez | Network Science Lead |
| Marc Jacob Doria | Genomics |
| Gerardo Luis Fernando | Genomics |

## Data Sources

| Source | Data |
|---|---|
| [ECDC Surveillance Atlas](https://atlas.ecdc.europa.eu) | AMR resistance rates (% resistant isolates) for 10 pathogen–antibiotic combinations, EU/EEA countries |
| [HadEX3 ETCCDI](https://www.climdex.org/access/) | Annual extreme temperature indices (TXx: hottest-day temperature) in NetCDF format |
| World Bank | Covariates: GDP per capita, population density, sanitation, water access |

## Pipeline

Run scripts in order from the project root directory.

```
python scripts/01_extract_hadex3.py
python scripts/02_preprocess.py
python scripts/03_network.py
python scripts/04_analysis.py
python scripts/05_network_final.py
python scripts/06_community.py
python scripts/07_community_network_overlay.py
python scripts/08_inter_community_bridges.py
python scripts/09_centrality_confounder_correlation.py
```

### Script descriptions

| Script | Description | Key outputs |
|---|---|---|
| `01_extract_hadex3.py` | Extracts country-level annual TXx means from HadEX3 NetCDF files using regionmask and cosine-latitude weighting | `data/processed/hadex3_country_annual.csv` |
| `02_preprocess.py` | Cleans ECDC AMR data (10 combos), harmonises country codes, merges with HadEX3 and World Bank covariates | `data/processed/master_dataset.csv`, `data/processed/ecdc_clean.csv`, `data/processed/covariates_clean.csv` |
| `03_network.py` | Bins countries into LOW/HIGH temperature groups (median TXx = 33.66 °C); builds 27×10 resistance matrix; constructs cosine-similarity network (threshold > 0.96); computes four centrality metrics | `data/processed/country_temp_groups.csv`, `data/processed/resistance_matrix.csv`, `data/processed/centrality_scores.csv`, `outputs/networks/amr_similarity_network.gexf` |
| `04_analysis.py` | Mann–Whitney U tests (LOW vs HIGH) on centrality metrics and resistance rate; centrality boxplots; network map; temperature vs resistance scatter with OLS regression | `data/processed/mannwhitney_results.csv`, `outputs/figures/centrality_boxplots.png`, `outputs/figures/amr_network_map.png`, `outputs/figures/temp_vs_resistance.png` |
| `05_network_final.py` | Publication-quality network figure coloured by temperature group | `outputs/figures/amr_network_final.png` |
| `06_community.py` | Louvain community detection (python-louvain, seed=42); 4 communities, modularity Q = 0.506 | `data/processed/community_assignments.csv`, `outputs/networks/amr_similarity_network.gexf` (updated with community attr) |
| `07_community_network_overlay.py` | Two-panel network comparison: left panel coloured by Louvain community, right panel coloured by temperature group | `outputs/figures/community_overlay_comparison.png`, `outputs/processed/community_composition.csv` |
| `08_inter_community_bridges.py` | Participation coefficient (Guimera & Amaral 2005) per node; classifies nodes into Connector hub / Kinless / Provincial hub / Peripheral; scatter plot of betweenness vs participation | `outputs/figures/bridge_participation.png`, `outputs/processed/bridge_analysis.csv` |
| `09_centrality_confounder_correlation.py` | OLS regression (mean resistance ~ TXx); per-country residuals; Spearman correlation of residuals vs each centrality metric; outlier table | `outputs/figures/residual_centrality_correlation.png`, `outputs/figures/residual_outlier_table.png`, `outputs/processed/regression_residuals.csv` |

## Key Results

- **27 EU/EEA countries**, 2000–2018, 10 pathogen–antibiotic combinations from ECDC surveillance
- **Network:** 27 nodes, 66 edges, density = 0.188, cosine similarity threshold > 0.96, fully connected
- **Temperature split:** LOW (n = 13, TXx < 33.66 °C) vs HIGH (n = 14, TXx ≥ 33.66 °C)
- **Significant findings (Mann–Whitney U, two-sided):**
  - Resistance rate: LOW median = 15.4%, HIGH median = 29.5% (U = 26.0, p = 0.0017)
  - Eigenvector centrality: LOW median = 0.232, HIGH median ≈ 0.000 (U = 133.0, p = 0.044)
- **OLS regression** (TXx → mean resistance): R² = 0.504, slope = 1.68%/°C, p < 0.001
- **Louvain communities:** 4 communities (Q = 0.506); Community 0 is the LOW-dominated western European hub

## Repository Structure

```
.
├── data/
│   ├── raw/                    # excluded from git (see .gitignore)
│   └── processed/              # excluded from git
├── docs/
│   ├── clark_personal_report.tex / .pdf   # personal class report
│   └── junior_handoff.tex / .pdf          # reference document for genomics juniors
├── outputs/
│   ├── figures/                # all PNG figures (excluded from git)
│   ├── networks/               # GEXF network file (tracked)
│   └── processed/              # extension analysis CSVs
├── scripts/                    # 01–09 analysis scripts
├── requirements.txt
└── sn-article.tex / .pdf       # Springer Nature article draft
```

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.10+. Raw data files (HadEX3 NetCDF, ECDC CSVs) must be placed in `data/raw/` before running scripts 01 and 02.

## Documents

- `docs/clark_personal_report.pdf` — personal narrative report (5 sections): pipeline overview, network interpretation, key findings, limitations, team roles
- `docs/junior_handoff.pdf` — technical reference for genomics collaborators: full country tables, MWU results, community breakdown, ready-to-paste Results paragraphs, discussion points
