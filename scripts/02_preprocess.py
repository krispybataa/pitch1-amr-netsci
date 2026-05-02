#!/usr/bin/env python3
"""
Step 2 — ECDC + covariates preprocessing and final merge.

2a: Load all 10 ECDC CSVs, filter to resistance-rate rows, harmonise to ISO3.
    Output: data/processed/ecdc_clean.csv

2b: Load 4 covariate CSVs (GDP, population density, sanitation, water),
    filter to 2000-2018, merge into one frame.
    Output: data/processed/covariates_clean.csv

2c: Three-way merge of HadEX3 + ECDC + covariates.
    Output: data/processed/master_dataset.csv
"""

import glob
import os
from pathlib import Path

import pandas as pd
import pycountry

RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")
PROC_DIR.mkdir(parents=True, exist_ok=True)

YEAR_MIN, YEAR_MAX = 2000, 2018

# ECDC uses EU/NUTS codes that differ from ISO 3166-1 alpha-2
_ECDC_CODE_OVERRIDES: dict[str, str] = {
    "EL": "GRC",  # Greece  (ECDC uses EL, ISO uses GR)
    "UK": "GBR",  # United Kingdom (ECDC uses UK, ISO uses GB)
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def alpha2_to_alpha3(a2: str) -> str | None:
    if a2 in _ECDC_CODE_OVERRIDES:
        return _ECDC_CODE_OVERRIDES[a2]
    c = pycountry.countries.get(alpha_2=a2)
    return c.alpha_3 if c else None


# ---------------------------------------------------------------------------
# Step 2a — ECDC
# ---------------------------------------------------------------------------

# Indicator string as it appears in the files (with trailing whitespace)
_TARGET_INDICATOR = "R - resistant isolates, percentage"


def process_ecdc() -> pd.DataFrame:
    files = sorted(glob.glob(str(RAW_DIR / "ecdc_*.csv")))
    if not files:
        raise FileNotFoundError(f"No ecdc_*.csv files found in {RAW_DIR}")

    chunks = []
    for f in files:
        df = pd.read_csv(f)

        # Filter to resistance-rate rows (strip whitespace before comparing)
        df = df[df["Indicator"].str.strip() == _TARGET_INDICATOR].copy()
        if df.empty:
            print(f"  WARNING: no resistance-rate rows in {os.path.basename(f)}")
            continue

        # Extract pathogen and antibiotic from "Pathogen|Antibiotic"
        split = df["Population"].str.split("|", n=1, expand=True)
        df["pathogen"] = split[0].str.strip()
        df["antibiotic"] = split[1].str.strip()

        # Keep and rename relevant columns
        df = df[["RegionCode", "RegionName", "Time", "pathogen", "antibiotic", "NumValue"]].copy()
        df = df.rename(columns={
            "RegionCode": "iso2",
            "RegionName": "country_name",
            "Time": "year",
            "NumValue": "resistance_rate",
        })
        # ECDC uses "-" as a sentinel for suppressed/unavailable values
        df["resistance_rate"] = pd.to_numeric(df["resistance_rate"], errors="coerce")

        chunks.append(df)

    ecdc = pd.concat(chunks, ignore_index=True)

    # Convert iso2 -> iso3
    ecdc["iso3"] = ecdc["iso2"].map(alpha2_to_alpha3)
    unresolved = ecdc[ecdc["iso3"].isna()]["iso2"].unique()
    if len(unresolved):
        print(f"  WARNING: could not resolve ISO3 for ECDC codes: {unresolved}")

    ecdc = ecdc.dropna(subset=["iso3"])

    # Filter years
    ecdc["year"] = ecdc["year"].astype(int)
    ecdc = ecdc[(ecdc["year"] >= YEAR_MIN) & (ecdc["year"] <= YEAR_MAX)]

    # Final column order
    ecdc = ecdc[["iso3", "country_name", "year", "pathogen", "antibiotic", "resistance_rate"]]
    ecdc = ecdc.sort_values(["iso3", "year", "pathogen", "antibiotic"]).reset_index(drop=True)

    out = PROC_DIR / "ecdc_clean.csv"
    ecdc.to_csv(out, index=False)
    print(f"  Saved {len(ecdc):,} rows -> {out}")
    print(f"  Countries: {ecdc['iso3'].nunique()}  Combos: {ecdc[['pathogen','antibiotic']].drop_duplicates().shape[0]}")
    return ecdc


# ---------------------------------------------------------------------------
# Step 2b — Covariates
# ---------------------------------------------------------------------------

# Maps filename fragment -> output column name
_COVARIATE_FILES: dict[str, str] = {
    "gdp-per-capita-worldbank.csv":                               "gdp_per_capita",
    "population-density.csv":                                     "pop_density",
    "share-of-population-with-improved-sanitation-faciltities.csv": "sanitation",
    "population-using-at-least-basic-drinking-water.csv":         "water",
}


def _load_covariate(fname: str, col_name: str) -> pd.DataFrame:
    path = RAW_DIR / fname
    df = pd.read_csv(path)

    # The value column is the one that isn't Entity, Code, or Year
    value_cols = [c for c in df.columns if c not in ("Entity", "Code", "Year")]
    if len(value_cols) != 1:
        # GDP has an extra 'World region' column — drop it first
        df = df.drop(columns=[c for c in value_cols if "region" in c.lower()], errors="ignore")
        value_cols = [c for c in df.columns if c not in ("Entity", "Code", "Year")]
    value_col = value_cols[0]

    df = df.rename(columns={"Code": "iso3", "Year": "year", value_col: col_name})
    df = df[["iso3", "year", col_name]].copy()

    # Filter years and drop rows with no ISO3
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["iso3", "year"])
    df = df[(df["year"] >= YEAR_MIN) & (df["year"] <= YEAR_MAX)]
    df["year"] = df["year"].astype(int)
    df = df.dropna(subset=[col_name])

    return df


def process_covariates() -> pd.DataFrame:
    frames = []
    for fname, col_name in _COVARIATE_FILES.items():
        df = _load_covariate(fname, col_name)
        print(f"  {col_name}: {len(df):,} rows, {df['iso3'].nunique()} countries")
        frames.append(df)

    # Sequential left-joins on iso3 + year; outer so we keep all country-years
    cov = frames[0]
    for df in frames[1:]:
        cov = cov.merge(df, on=["iso3", "year"], how="outer")

    cov = cov.sort_values(["iso3", "year"]).reset_index(drop=True)

    out = PROC_DIR / "covariates_clean.csv"
    cov.to_csv(out, index=False)
    print(f"  Saved {len(cov):,} rows -> {out}")
    return cov


# ---------------------------------------------------------------------------
# Step 2c — Final merge
# ---------------------------------------------------------------------------


def build_master(ecdc: pd.DataFrame, cov: pd.DataFrame) -> pd.DataFrame:
    hadex = pd.read_csv(PROC_DIR / "hadex3_country_annual.csv")
    hadex["year"] = hadex["year"].astype(int)

    # ECDC (left) x HadEX3 (right): inner join keeps only country-years
    # present in BOTH. ECDC is EU/EEA only; HadEX3 has global coverage.
    master = ecdc.merge(hadex, on=["iso3", "year"], how="inner")

    # Attach covariates: left join so climate rows without covariate data
    # are still represented (shown as NaN).
    master = master.merge(cov, on=["iso3", "year"], how="left")

    # Enforce final column order
    final_cols = [
        "iso3", "country_name", "year",
        "pathogen", "antibiotic", "resistance_rate",
        "TXx", "TNx", "SU", "TR", "WSDI", "TN90p",
        "gdp_per_capita", "pop_density", "sanitation", "water",
    ]
    master = master[final_cols]
    master = master.sort_values(["iso3", "year", "pathogen", "antibiotic"]).reset_index(drop=True)

    out = PROC_DIR / "master_dataset.csv"
    master.to_csv(out, index=False)

    # Summary
    print(f"\n{'='*60}")
    print(f"master_dataset.csv")
    print(f"  Shape:     {master.shape}")
    print(f"  Countries: {master['iso3'].nunique()}")
    print(f"  Years:     {master['year'].min()} - {master['year'].max()}")
    print(f"  Pathogens x Antibiotics: {master[['pathogen','antibiotic']].drop_duplicates().shape[0]} combos")
    print()
    null_counts = master.isnull().sum()
    null_pct = (null_counts / len(master) * 100).round(1)
    null_summary = pd.DataFrame({"null_count": null_counts, "null_pct": null_pct})
    null_summary = null_summary[null_summary["null_count"] > 0]
    if null_summary.empty:
        print("  No nulls.")
    else:
        print("  Null summary:")
        print(null_summary.to_string())
    print(f"{'='*60}")
    return master


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=== Step 2a: ECDC ===")
    ecdc = process_ecdc()

    print("\n=== Step 2b: Covariates ===")
    cov = process_covariates()

    print("\n=== Step 2c: Master merge ===")
    build_master(ecdc, cov)


if __name__ == "__main__":
    main()
