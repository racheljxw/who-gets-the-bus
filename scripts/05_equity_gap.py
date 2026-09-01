"""
Phase 3 -- merge access_score + need_score -> equity_gap (the headline layer).

Inputs:
  data/processed/access.parquet   (Phase 1b, 158 rows)
  data/processed/need.parquet     (Phase 2, 158 rows)
  data/raw/neighbourhoods-4326.geojson   (158 polygons, EPSG:4326 -- geometry only)

Steps (CLAUDE.md section 5):
  1. Merge access + need on AREA_SHORT_CODE (inner). Both carry all 158
     neighbourhoods; assert the merged row count is EXACTLY 158 and fail loudly
     otherwise -- anything else means a join-key mismatch we have not seen.
  2. equity_gap = need_score - access_score   (range ~ -1 .. +1). Print min/max.
  3. Merge the scored columns onto the raw neighbourhood polygons (keyed by
     AREA_SHORT_CODE) so one GeoDataFrame carries geometry + all scores + the
     raw tooltip inputs from both sides.
  4. Write:
       data/processed/equity.parquet    flat table, no geometry (CSV/XLSX export)
       data/processed/equity.geojson    same columns + geometry (Phase 4 map)
  5. Report: 10 highest equity_gap (most underserved) + 10 lowest (most
     over-served relative to need), and the "well-matched" neighbourhoods where
     BOTH scores sit within 0.05 of their respective extremes (high need + high
     access, or low need + low access) -- interesting cases that never surface at
     the ends of the gap ranking.

NOTE: both source parquets have a column literally named "density_norm" but they
mean different things (access = stop density, need = population density). Neither
is needed downstream, so we select explicit column lists and never merge them.

Usage:
    python scripts/05_equity_gap.py
"""

from __future__ import annotations

import os
import sys

import geopandas as gpd
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

HERE = os.path.dirname(__file__)
ACCESS_PARQUET = os.path.abspath(os.path.join(HERE, "..", "data", "processed", "access.parquet"))
NEED_PARQUET = os.path.abspath(os.path.join(HERE, "..", "data", "processed", "need.parquet"))
GEOJSON = os.path.abspath(os.path.join(HERE, "..", "data", "raw", "neighbourhoods-4326.geojson"))
OUT_PARQUET = os.path.abspath(os.path.join(HERE, "..", "data", "processed", "equity.parquet"))
OUT_GEOJSON = os.path.abspath(os.path.join(HERE, "..", "data", "processed", "equity.geojson"))

# Columns carried into both outputs (per Phase 3 spec).
OUT_COLS = [
    "AREA_SHORT_CODE", "AREA_NAME", "access_score", "need_score", "equity_gap",
    "stop_count", "neighbourhood_frequency", "neighbourhood_stop_density",
    "low_income_pct", "non_car_commute_pct", "population_density", "imputed_flag",
]

MATCH_TOL = 0.05  # "within 0.05 of the extreme" for the well-matched flag


def main() -> int:
    for p in (ACCESS_PARQUET, NEED_PARQUET, GEOJSON):
        if not os.path.isfile(p):
            print(f"missing {p}")
            return 1

    print("=" * 78)
    print("Phase 3: merge access + need -> equity_gap")
    print("=" * 78)

    # ---- 1. merge access + need on AREA_SHORT_CODE ----
    access = pd.read_parquet(ACCESS_PARQUET)[[
        "AREA_SHORT_CODE", "AREA_NAME", "access_score", "stop_count",
        "neighbourhood_frequency", "neighbourhood_stop_density",
    ]].copy()
    need = pd.read_parquet(NEED_PARQUET)[[
        "AREA_SHORT_CODE", "AREA_NAME", "need_score", "low_income_pct",
        "non_car_commute_pct", "population_density", "imputed_flag",
    ]].copy()
    access["AREA_SHORT_CODE"] = access["AREA_SHORT_CODE"].astype(int)
    need["AREA_SHORT_CODE"] = need["AREA_SHORT_CODE"].astype(int)

    print(f"  access.parquet: {len(access)} rows   need.parquet: {len(need)} rows")

    merged = access.merge(
        need.drop(columns="AREA_NAME"), on="AREA_SHORT_CODE", how="inner"
    )
    if len(merged) != 158:
        raise SystemExit(
            f"FATAL: merged row count is {len(merged)}, expected exactly 158. "
            f"access codes not in need: "
            f"{sorted(set(access['AREA_SHORT_CODE']) - set(need['AREA_SHORT_CODE']))}; "
            f"need codes not in access: "
            f"{sorted(set(need['AREA_SHORT_CODE']) - set(access['AREA_SHORT_CODE']))}. "
            f"This is a join-key mismatch we have not seen before -- investigate."
        )
    assert merged["AREA_SHORT_CODE"].is_unique, "AREA_SHORT_CODE not unique after merge"
    print(f"  merged on AREA_SHORT_CODE (inner): {len(merged)} rows -- all 158 matched 1:1")

    # ---- 2. equity_gap = need - access ----
    merged["equity_gap"] = merged["need_score"] - merged["access_score"]
    gmin, gmax = merged["equity_gap"].min(), merged["equity_gap"].max()
    print(f"\n  equity_gap = need_score - access_score")
    print(f"  equity_gap range: {gmin:.4f} .. {gmax:.4f}  (expected roughly -1 .. +1)")
    assert -1.0 <= gmin and gmax <= 1.0, "equity_gap outside [-1, 1] -- unexpected"

    # ---- 3. merge scored columns onto the raw polygons ----
    geo = gpd.read_file(GEOJSON)[["AREA_SHORT_CODE", "AREA_NAME", "geometry"]].copy()
    geo["AREA_SHORT_CODE"] = geo["AREA_SHORT_CODE"].astype(int)
    assert len(geo) == 158, f"geojson has {len(geo)} features, expected 158"

    gdf = geo.drop(columns="AREA_NAME").merge(
        merged[OUT_COLS], on="AREA_SHORT_CODE", how="inner"
    )
    if len(gdf) != 158:
        raise SystemExit(
            f"FATAL: geometry join produced {len(gdf)} rows, expected 158 -- "
            f"AREA_SHORT_CODE mismatch between geojson and the scored table."
        )
    gdf = gdf[OUT_COLS + ["geometry"]]
    assert gdf.crs is not None and gdf.crs.to_epsg() == 4326, f"expected EPSG:4326, got {gdf.crs}"
    print(f"  joined scores onto polygons: {len(gdf)} features, CRS {gdf.crs.to_epsg()}")

    # ---- 4. write both outputs ----
    flat = merged[OUT_COLS].sort_values("equity_gap", ascending=False).reset_index(drop=True)
    os.makedirs(os.path.dirname(OUT_PARQUET), exist_ok=True)
    flat.to_parquet(OUT_PARQUET, index=False)
    if os.path.exists(OUT_GEOJSON):
        os.remove(OUT_GEOJSON)
    gdf.to_file(OUT_GEOJSON, driver="GeoJSON")
    print(f"\n  wrote {OUT_PARQUET}  ({len(flat)} rows)")
    print(f"  wrote {OUT_GEOJSON}  ({len(gdf)} features)")

    # ---- 5. report ----
    def _row(r):
        return (f"    gap={r['equity_gap']:+.3f}  {r['AREA_NAME']:<38} "
                f"need={r['need_score']:.3f}  access={r['access_score']:.3f}")

    print("\n" + "-" * 78)
    print("  TOP 10 equity_gap -- most UNDERSERVED (high need, low access):")
    for _, r in flat.head(10).iterrows():
        print(_row(r))

    print("\n  BOTTOM 10 equity_gap -- most OVER-SERVED relative to need:")
    for _, r in flat.tail(10).iloc[::-1].iterrows():
        print(_row(r))

    # well-matched: both scores within MATCH_TOL of their own extreme
    a_hi, a_lo = merged["access_score"].max(), merged["access_score"].min()
    n_hi, n_lo = merged["need_score"].max(), merged["need_score"].min()
    high_high = merged[
        (merged["access_score"] >= a_hi - MATCH_TOL)
        & (merged["need_score"] >= n_hi - MATCH_TOL)
    ]
    low_low = merged[
        (merged["access_score"] <= a_lo + MATCH_TOL)
        & (merged["need_score"] <= n_lo + MATCH_TOL)
    ]
    print(f"\n  WELL-MATCHED cases (both scores within {MATCH_TOL} of their extreme --")
    print(f"  won't show at the ends of the gap ranking, but worth knowing):")
    print(f"    access_score extremes: {a_lo:.3f} .. {a_hi:.3f}   "
          f"need_score extremes: {n_lo:.3f} .. {n_hi:.3f}")
    print(f"\n    HIGH need + HIGH access ({len(high_high)}):")
    for _, r in high_high.sort_values("need_score", ascending=False).iterrows():
        print(f"      {r['AREA_NAME']:<38} need={r['need_score']:.3f}  "
              f"access={r['access_score']:.3f}  gap={r['equity_gap']:+.3f}")
    print(f"\n    LOW need + LOW access ({len(low_low)}):")
    for _, r in low_low.sort_values("need_score").iterrows():
        print(f"      {r['AREA_NAME']:<38} need={r['need_score']:.3f}  "
              f"access={r['access_score']:.3f}  gap={r['equity_gap']:+.3f}")

    print(f"\n  imputed_flag True for {int(merged['imputed_flag'].sum())} neighbourhood(s)")
    print("\nPhase 3 done. Next: Phase 4 map build (needs confirmation).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
