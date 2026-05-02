# Pitch 1 — AMR Network Topology & Climate Change

**Research Question:** Do higher temperatures increase the network connectivity
of AMR transmission? Which countries/pathogens act as bridge nodes under extreme heat?

## Team
- Clark Rodriguez (NetSci Lead)
- Marc Jacob Doria (Genomics)
- Gerardo Luis Fernando (Genomics)

## Pipeline Phases
1. Data Acquisition (ResistanceMap, NOAA ETCCDI, World Bank)
2. Preprocessing & Harmonization (ISO 3166-1 alpha-3, 2010–2020 window)
3. Network Construction (bipartite → co-occurrence/similarity network)
4. Centrality Analysis (degree, betweenness, closeness, eigenvector)
5. Statistical Comparison (Mann-Whitney U across temp quantiles)
6. Visualization (Gephi)

## Data Sources
- https://resistancemap.onehealthtrust.org
- https://www.ncei.noaa.gov (ETCCDI or GST)
- https://data.worldbank.org