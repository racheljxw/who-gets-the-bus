"""
Phase 2 -- demographics -> neighbourhood need_score.

Input:
  data/raw/nbhd_2021_census_profile_full_158model.xlsx   sheet hd2021_census_profile
  data/processed/access.parquet                          (Phase 1b -- for area_km2)

The profile sheet is TRANSPOSED (recon, 2026-09-01): column 0 is the characteristic
label, columns 1..158 are one neighbourhood each. Read with header=None and address
source values by ROW INDEX:
  row 0    -> Neighbourhood Name
  row 1    -> Neighbourhood Number   (join key; int, 1..174 with gaps, 158 values)
  row 3    -> "Total - Age groups of the population - 25% sample data"  (population)
  row 178  -> "Prevalence of low income based on the Low-income measure, after tax
              (LIM-AT) (%)"
  row 2575 -> "Total - Main mode of commuting for the employed labour force ..."
  row 2576 -> "  Car, truck or van"

CONFIRMED PROXY DECISIONS (human-confirmed 2026-09-01 -- deviations from the CLAUDE.md
section 5 kickoff spec, documented there too):

  1. low_income_pct = row 178 (LIM-AT prevalence %). The spec asked for
     "% low-income households"; the 2021 Census profile has NO household-level
     low-income variable. LIM-AT prevalence is persons-in-private-households based
     and is the standard Toronto neighbourhood low-income indicator.

  2. non_car_commute_pct = (row 2575 - row 2576) / row 2575 * 100
     i.e. (all commuters - car/truck/van commuters) / all commuters.
     The spec asked for "% households with no vehicle available"; the 2021 Census
     dropped the household vehicle-ownership question, so no direct match exists.
     "Main mode of commuting" is the only transport block. We use the non-car share
     (walk + bike + transit + other) rather than transit-only, because it captures
     all car-independent commuters -- closer to the spirit of "no vehicle access".
     Note: base is the employed labour force with a workplace, not households.

  3. population_density = row 3 population / area_km2, where area_km2 comes from
     data/processed/access.parquet (our own EPSG:2952 polygon areas from Phase 1b).
     The profile ships no land-area or density figure; reusing the Phase 1 area
     keeps the two scores geometrically consistent.

Scoring (CLAUDE.md section 5, need_score / raw_need -- unchanged):
  low_income_norm = min-max(low_income_pct)          -> 0..1
  non_car_norm    = min-max(non_car_commute_pct)     -> 0..1
  density_norm    = min-max(population_density)      -> 0..1
  need_score      = mean(the three norms)            -> 0..1

Edge case (CLAUDE.md section 5): suppressed census cells -> impute with the citywide
median of that variable AND flag the neighbourhood (imputed_flag = True if ANY of the
three raw inputs was imputed). Recon found 0 suppressed cells in the 3 source rows;
this re-validates rather than assumes.

Output:
  data/processed/need.parquet -- AREA_SHORT_CODE, AREA_NAME, low_income_pct,
  non_car_commute_pct, population, population_density, low_income_norm, non_car_norm,
  density_norm, need_score, imputed_flag

Phase 3 (merge -> equity_gap) is next and needs explicit confirmation.

Usage:
    python scripts/04_need_score.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

HERE = os.path.dirname(__file__)
PROFILE_XLSX = os.path.abspath(
    os.path.join(HERE, "..", "data", "raw", "nbhd_2021_census_profile_full_158model.xlsx")
)
ACCESS_PARQUET = os.path.abspath(
    os.path.join(HERE, "..", "data", "processed", "access.parquet")
)
OUT_PARQUET = os.path.abspath(
    os.path.join(HERE, "..", "data", "processed", "need.parquet")
)

SHEET = "hd2021_census_profile"

# Source row indices on the transposed sheet (header=None). See module docstring.
ROW_NAME = 0
ROW_NUMBER = 1
ROW_POPULATION = 3
ROW_LOW_INCOME = 178
ROW_COMMUTE_TOTAL = 2575
ROW_COMMUTE_CAR = 2576

# Expected label text at each row -- asserted so a re-published profile that shifts
# rows fails loudly instead of silently scoring the wrong variable.
EXPECTED_LABELS = {
    ROW_NAME: "Neighbourhood Name",
    ROW_NUMBER: "Neighbourhood Number",
    ROW_POPULATION: "Total - Age groups of the population - 25% sample data",
    ROW_LOW_INCOME: (
        "Prevalence of low income based on the Low-income measure, after tax "
        "(LIM-AT) (%)"
    ),
    ROW_COMMUTE_TOTAL: (
        "Total - Main mode of commuting for the employed labour force aged 15 "
        "years and over with a usual place of work or no fixed workplace address "
        "- 25% sample data"
    ),
    ROW_COMMUTE_CAR: "Car, truck or van",
}


def minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if hi == lo:
        return pd.Series(0.0, index=s.index)
    return (s - lo) / (hi - lo)


def row_values(df: pd.DataFrame, row_idx: int) -> pd.Series:
    """Neighbourhood values (cols 1..158) for one characteristic row, as numeric."""
    return pd.to_numeric(df.iloc[row_idx, 1:], errors="coerce").reset_index(drop=True)


def main() -> int:
    for p in (PROFILE_XLSX, ACCESS_PARQUET):
        if not os.path.isfile(p):
            print(f"missing {p}")
            return 1

    print("=" * 78)
    print("Phase 2: demographics -> need_score")
    print("=" * 78)

    # ---- 1. load transposed profile sheet ----
    prof = pd.read_excel(PROFILE_XLSX, sheet_name=SHEET, header=None, engine="openpyxl")
    print(f"  loaded {SHEET}: {prof.shape[0]} rows x {prof.shape[1]} cols "
          f"({prof.shape[1] - 1} neighbourhood columns)")

    for idx, want in EXPECTED_LABELS.items():
        got = str(prof.iat[idx, 0]).strip()
        assert got == want, (
            f"row {idx} label mismatch -- expected {want!r}, got {got!r}. "
            f"The profile layout changed; re-run recon before trusting this script."
        )
    print("  row-label assertions passed (source rows still where recon found them)")

    # ---- 2. extract the three raw inputs, keyed by Neighbourhood Number ----
    numbers = pd.to_numeric(prof.iloc[ROW_NUMBER, 1:], errors="coerce").astype("Int64")
    names = prof.iloc[ROW_NAME, 1:].astype(str).reset_index(drop=True)
    numbers = numbers.reset_index(drop=True)
    assert numbers.notna().all(), "some Neighbourhood Number cells are blank"
    assert numbers.is_unique, "Neighbourhood Number is not unique across columns"

    population = row_values(prof, ROW_POPULATION)
    low_income_pct = row_values(prof, ROW_LOW_INCOME)
    commute_total = row_values(prof, ROW_COMMUTE_TOTAL)
    commute_car = row_values(prof, ROW_COMMUTE_CAR)
    non_car_commute_pct = (commute_total - commute_car) / commute_total * 100

    demo = pd.DataFrame({
        "AREA_SHORT_CODE": numbers.astype(int),
        "profile_name": names,
        "low_income_pct": low_income_pct,
        "non_car_commute_pct": non_car_commute_pct,
        "population": population,
    })
    print(f"  extracted {len(demo)} neighbourhoods from the profile")

    # ---- 3. join Phase 1b area_km2, derive population_density ----
    access = pd.read_parquet(ACCESS_PARQUET)[["AREA_SHORT_CODE", "AREA_NAME", "area_km2"]]
    access["AREA_SHORT_CODE"] = access["AREA_SHORT_CODE"].astype(int)

    need = access.merge(demo, on="AREA_SHORT_CODE", how="left")
    missing_join = need[need["profile_name"].isna()]
    assert missing_join.empty, (
        f"{len(missing_join)} neighbourhoods in access.parquet had no profile row: "
        f"{missing_join['AREA_SHORT_CODE'].tolist()}"
    )
    assert len(need) == 158, f"expected 158 neighbourhoods, got {len(need)}"
    print(f"  joined to access.parquet on AREA_SHORT_CODE: {len(need)} rows, all matched")

    need["population_density"] = need["population"] / need["area_km2"]

    # ---- 4. suppressed-cell check -> impute citywide median + flag ----
    RAW_INPUTS = ["low_income_pct", "non_car_commute_pct", "population_density"]
    need["imputed_flag"] = False
    print("\n  suppressed / null check on the three raw inputs:")
    for col in RAW_INPUTS:
        null_mask = need[col].isna()
        n_null = int(null_mask.sum())
        print(f"    {col:<22}: {n_null} null")
        if n_null:
            med = need[col].median()
            for _, r in need[null_mask].iterrows():
                print(f"      imputed {r['AREA_NAME']} (code {r['AREA_SHORT_CODE']}) "
                      f"-> citywide median {med:.4f}")
            need.loc[null_mask, col] = med
            need.loc[null_mask, "imputed_flag"] = True
    n_imputed = int(need["imputed_flag"].sum())
    print(f"  imputed_flag True for {n_imputed} neighbourhood(s)")

    # ---- 5. normalize each term, then average (CLAUDE.md section 5) ----
    need["low_income_norm"] = minmax(need["low_income_pct"])
    need["non_car_norm"] = minmax(need["non_car_commute_pct"])
    need["density_norm"] = minmax(need["population_density"])
    need["need_score"] = need[
        ["low_income_norm", "non_car_norm", "density_norm"]
    ].mean(axis=1)

    # ---- 6. write ----
    need = need[[
        "AREA_SHORT_CODE", "AREA_NAME", "low_income_pct", "non_car_commute_pct",
        "population", "population_density", "low_income_norm", "non_car_norm",
        "density_norm", "need_score", "imputed_flag",
    ]].sort_values("need_score", ascending=False).reset_index(drop=True)

    os.makedirs(os.path.dirname(OUT_PARQUET), exist_ok=True)
    need.to_parquet(OUT_PARQUET, index=False)

    # ---- 7. report ----
    print("\n" + "-" * 78)
    print(f"  wrote {OUT_PARQUET}  ({len(need)} rows)")
    print(f"  imputed rows: {n_imputed}")

    print("\n  raw input ranges (sanity check):")
    print(f"    low_income_pct       : {need['low_income_pct'].min():6.2f} .. "
          f"{need['low_income_pct'].max():6.2f}  (median {need['low_income_pct'].median():.2f})  [LIM-AT %]")
    print(f"    non_car_commute_pct  : {need['non_car_commute_pct'].min():6.2f} .. "
          f"{need['non_car_commute_pct'].max():6.2f}  (median {need['non_car_commute_pct'].median():.2f})  [%]")
    print(f"    population_density   : {need['population_density'].min():6.0f} .. "
          f"{need['population_density'].max():6.0f}  (median {need['population_density'].median():.0f})  [people/km^2]")

    def _row(r):
        return (f"    {r['need_score']:.3f}  {r['AREA_NAME']:<38} "
                f"LIM-AT={r['low_income_pct']:5.1f}% (n={r['low_income_norm']:.2f})  "
                f"non-car={r['non_car_commute_pct']:5.1f}% (n={r['non_car_norm']:.2f})  "
                f"dens={r['population_density']:6.0f} (n={r['density_norm']:.2f})")

    print("\n  TOP 5 need_score:")
    for _, r in need.head(5).iterrows():
        print(_row(r))
    print("\n  BOTTOM 5 need_score:")
    for _, r in need.tail(5).iterrows():
        print(_row(r))

    print(f"\n  need_score range: {need['need_score'].min():.4f} .. {need['need_score'].max():.4f}")
    print("\nPhase 2 done. Next: Phase 3 merge -> equity_gap (needs confirmation).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
