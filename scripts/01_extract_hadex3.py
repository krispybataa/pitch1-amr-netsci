#!/usr/bin/env python3
"""
Step 1 — Extract country-level annual means from HadEX3 NetCDF files.

For each of the six ETCCDI indices (TXx, TNx, SU, TR, WSDI, TN90p), this
script:
  1. Decompresses the .nc.gz to a temp file.
  2. Loads it with xarray and slices to 2000-2018.
  3. Applies Natural Earth 110m country polygons (via regionmask) as a mask.
  4. Computes a cosine-latitude-weighted spatial mean for every country × year.
  5. Merges all six indices into a single wide DataFrame.

Output: data/processed/hadex3_country_annual.csv
Columns: iso3, year, TXx, TNx, SU, TR, WSDI, TN90p
"""

import gzip
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pycountry
import regionmask
import xarray as xr
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

YEAR_MIN, YEAR_MAX = 2000, 2018

HADEX_FILES: dict[str, str] = {
    "TXx":   "HadEX3_TXx_1901-2018_ADW_61-90_1.25x1.875deg.nc.gz",
    "TNx":   "HadEX3_TNx_1901-2018_ADW_61-90_1.25x1.875deg.nc.gz",
    "SU":    "HadEX3_SU_1901-2018_ADW_61-90_1.25x1.875deg.nc.gz",
    "TR":    "HadEX3_TR_1901-2018_ADW_61-90_1.25x1.875deg.nc.gz",
    "WSDI":  "HadEX3_WSDI_1901-2018_ADW_61-90_1.25x1.875deg.nc.gz",
    "TN90p": "HadEX3_TN90p_1901-2018_ADW_61-90_1.25x1.875deg.nc.gz",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# Natural Earth 110m uses non-standard, abbreviated, or shared alpha-2 codes.
# We resolve by region *name* first (most reliable), then fall back to alpha-2.
#
# Key Natural Earth quirks (regionmask 0.13, natural_earth_v5_0_0.countries_110):
#  • Single-letter codes: E=Spain, F=France, I=Italy, A=Austria, B=Belgium,
#    D=Germany, L=Luxembourg, N=Norway, P=Portugal, S=Sweden
#  • Alpha-3-ish codes: ARM, DRC, EST, FIN, IND, INDO, IRL, IRN, IRQ, RUS, SYR
#  • Collision — same code for multiple regions:
#      "IS" → Iceland AND Israel  (pycountry picks Iceland)
#      "J"  → Jamaica, Japan, Jordan  (pycountry returns None)
#      "SL" → Sierra Leone AND Somaliland (pycountry picks Sierra Leone)
#      "CN" → China AND N. Cyprus (pycountry picks China)
#      "ES" → eSwatini in NE but pycountry resolves it to Spain!
#             (eSwatini ISO alpha-2 is SZ, not ES)
#  • Non-ISO entities: N. Cyprus, Somaliland → None (exclude)
_NAME_TO_ISO3: dict[str, str | None] = {
    # --- Single-letter NE codes (pycountry returns None) ---
    "Austria":              "AUT",
    "Belgium":              "BEL",
    "France":               "FRA",
    "Germany":              "DEU",
    "Italy":                "ITA",
    "Luxembourg":           "LUX",
    "Norway":               "NOR",
    "Portugal":             "PRT",
    "Spain":                "ESP",
    "Sweden":               "SWE",
    # --- Alpha-3-ish codes (pycountry returns None) ---
    "Armenia":              "ARM",
    "Dem. Rep. Congo":      "COD",
    "Estonia":              "EST",
    "Finland":              "FIN",
    "India":                "IND",
    "Indonesia":            "IDN",
    "Iran":                 "IRN",
    "Iraq":                 "IRQ",
    "Ireland":              "IRL",
    "North Macedonia":      "MKD",
    "Russia":               "RUS",
    "Slovenia":             "SVN",
    "Syria":                "SYR",
    # --- Non-standard short codes (pycountry returns None) ---
    "Bosnia and Herz.":     "BIH",
    "Kosovo":               "XKX",
    "Palestine":            "PSE",
    # --- Code-collision: "IS" shared by Iceland + Israel ---
    "Iceland":              "ISL",
    "Israel":               "ISR",
    # --- Code-collision: "J" shared by Jamaica, Japan, Jordan ---
    "Jamaica":              "JAM",
    "Japan":                "JPN",
    "Jordan":               "JOR",
    # --- Code-collision: "ES" in NE = eSwatini (ISO SZ), not Spain ---
    "eSwatini":             "SWZ",
    # --- Non-ISO / disputed territories — exclude ---
    "N. Cyprus":            None,
    "Somaliland":           None,
}

# Supplemental alpha-2 → alpha-3 overrides for codes pycountry doesn't know
_ALPHA2_OVERRIDES: dict[str, str] = {
    "XK": "XKX",  # Kosovo
    "PS": "PSE",  # Palestine
}


def get_iso3_for_region(region) -> str | None:
    """Return ISO 3166-1 alpha-3 for a regionmask _OneRegion; None to skip."""
    # Name-based lookup takes priority to handle Natural Earth code collisions
    if region.name in _NAME_TO_ISO3:
        return _NAME_TO_ISO3[region.name]
    a2 = region.abbrev
    if a2 in _ALPHA2_OVERRIDES:
        return _ALPHA2_OVERRIDES[a2]
    c = pycountry.countries.get(alpha_2=a2)
    return c.alpha_3 if c else None


def decompress_gz(gz_path: Path) -> Path:
    """Decompress a .nc.gz to a NamedTemporaryFile and return its path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".nc", delete=False)
    with gzip.open(gz_path, "rb") as src:
        shutil.copyfileobj(src, tmp)
    tmp.close()
    return Path(tmp.name)


def _find_data_var(ds: xr.Dataset, preferred: str) -> str:
    """Return the variable name to extract.

    HadEX3 NetCDF files store the index value as separate monthly (Jan–Dec)
    and annual ('Ann') variables. We always want 'Ann' for the annual series.
    Falls back to an exact or case-insensitive match on `preferred`, then to
    the single non-bounds variable if there is one.
    """
    if "Ann" in ds:
        return "Ann"
    if preferred in ds:
        return preferred
    for v in ds.data_vars:
        if v.lower() == preferred.lower():
            return v
    candidates = [v for v in ds.data_vars if "bnd" not in v.lower()]
    if len(candidates) == 1:
        return candidates[0]
    raise KeyError(
        f"Cannot identify variable '{preferred}' in dataset with vars {list(ds.data_vars)}"
    )


def _standardize_coords(da: xr.DataArray) -> xr.DataArray:
    """Rename latitude/longitude coordinate variants to 'lat'/'lon'."""
    rename = {}
    for c in list(da.coords) + list(da.dims):
        cl = c.lower()
        if cl == "latitude" and c != "lat":
            rename[c] = "lat"
        elif cl == "longitude" and c != "lon":
            rename[c] = "lon"
    return da.rename(rename) if rename else da


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------


def extract_variable(
    var_name: str,
    gz_path: Path,
    countries: regionmask.Regions,
) -> pd.DataFrame:
    """
    Return a DataFrame with columns [iso3, year, <var_name>] for all
    country-year combinations that have valid data in 2000-2018.
    """
    print(f"\n[{var_name}] Decompressing {gz_path.name} ...")
    nc_path = decompress_gz(gz_path)

    try:
        ds = xr.open_dataset(nc_path)
        vname = _find_data_var(ds, var_name)
        da = _standardize_coords(ds[vname])

        # HadEX3 time axis is plain integer years (e.g. 1901 … 2018)
        year_mask = (da.time >= YEAR_MIN) & (da.time <= YEAR_MAX)
        da = da.isel(time=year_mask)
        years = da.time.values.astype(int)

        if len(years) == 0:
            raise ValueError(f"No data in {YEAR_MIN}-{YEAR_MAX} for {var_name}")

        lat = da["lat"].values
        lon = da["lon"].values

        # Cosine-latitude weights — shape (lat,); broadcasts over lon/time
        cos_lat = xr.DataArray(np.cos(np.deg2rad(lat)), dims=["lat"])

        # Country mask: shape (region, lat, lon), boolean.
        # The 'region' coordinate holds region *numbers*, not 0-N indices.
        # Only regions with ≥1 grid cell appear in the output.
        mask3d = countries.mask_3D(lon, lat)  # (region, lat, lon)

        # countries.regions is already a dict: number → _OneRegion
        num_to_region = countries.regions

        rows: list[dict] = []

        for region_num in tqdm(
            mask3d.region.values, desc=f"  {var_name} countries", leave=False
        ):
            region = num_to_region[int(region_num)]
            iso3 = get_iso3_for_region(region)
            if iso3 is None:
                continue  # non-ISO or disputed territory; skip

            cell_mask = mask3d.sel(region=region_num)  # (lat, lon), boolean
            if not cell_mask.any():
                continue

            # Apply mask: NaN outside country, keep (time, lat, lon) shape
            da_c = da.where(cell_mask)

            # Weighted mean: Σ(val × cos_lat) / Σ(valid_cos_lat)
            numerator   = (da_c * cos_lat).sum(dim=["lat", "lon"])       # (time,)
            denominator = (da_c.notnull() * cos_lat).sum(dim=["lat", "lon"])  # (time,)
            means = (numerator / denominator).values  # (time,)

            for yr, val in zip(years, means):
                if not np.isnan(val):
                    rows.append({"iso3": iso3, "year": int(yr), var_name: float(val)})

        df = pd.DataFrame(rows)
        print(f"  -> {len(df):,} country-year records for {var_name}")
        return df

    finally:
        ds.close()
        nc_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # Natural Earth 110m polygons — coarse enough for a 1.25×1.875° grid,
    # abbrevs are ISO 3166-1 alpha-3 codes.
    countries = regionmask.defined_regions.natural_earth_v5_0_0.countries_110
    print(f"Loaded {len(countries)} country regions from Natural Earth 110m.")

    variable_dfs: list[pd.DataFrame] = []

    for var_name, filename in HADEX_FILES.items():
        gz_path = RAW_DIR / filename
        if not gz_path.exists():
            print(f"WARNING: {gz_path} not found — skipping {var_name}.")
            continue
        df = extract_variable(var_name, gz_path, countries)
        # Guard: if two Natural Earth regions still resolve to the same ISO3
        # (e.g. due to unlisted name overrides), take the mean so the merge
        # below doesn't produce a Cartesian product.
        var_col = [c for c in df.columns if c not in ("iso3", "year")]
        if df.duplicated(subset=["iso3", "year"]).any():
            dups = df[df.duplicated(subset=["iso3","year"],keep=False)]["iso3"].unique()
            print(f"  WARNING: duplicate (iso3, year) for {dups} — averaging.")
            df = df.groupby(["iso3", "year"], as_index=False)[var_col].mean()
        variable_dfs.append(df)

    if not variable_dfs:
        raise RuntimeError("No HadEX3 files were processed.")

    # Outer-join all variables on (iso3, year) so partial coverage is retained
    merged = variable_dfs[0]
    for df in variable_dfs[1:]:
        merged = merged.merge(df, on=["iso3", "year"], how="outer")

    merged = (
        merged
        .sort_values(["iso3", "year"])
        .reset_index(drop=True)
    )

    out_path = OUT_DIR / "hadex3_country_annual.csv"
    merged.to_csv(out_path, index=False)

    print(f"\n{'='*60}")
    print(f"Saved {len(merged):,} rows x {len(merged.columns)} cols -> {out_path}")
    print(f"Countries: {merged['iso3'].nunique()}  |  Years: {sorted(merged['year'].unique())}")
    print("\nSample:")
    print(merged.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
