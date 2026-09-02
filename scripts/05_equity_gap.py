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

# access_score / equity_gap after PATCH 1 (capacity weighting only: subway x15,
# streetcar x2.5, bus x1; freq_norm = minmax(own capacity-weighted frequency)),
# for the neighbourhoods flagged in the North Toronto investigation. Captured
# 2026-09-01 from data/processed/equity.parquet just before PATCH 2 (Fix A: split
# streetcar/LRT + add LRT x6; Fix B: 500 m rapid-transit walkshed credit) re-ran
# the pipeline. Kept inline so the patch-1 -> patch-2 diff is self-contained.
FLAGGED_PATCH1 = {
    "North Toronto":       {"access": 0.260794, "gap": 0.464267},
    "Yonge-Doris":         {"access": 0.283600, "gap": 0.439400},
    "Church-Wellesley":    {"access": 0.438200, "gap": 0.420800},
    "North St.James Town": {"access": 0.404100, "gap": 0.463300},
}

# access_score / equity_gap after PATCH 2 (per-PLATFORM walkshed credit), captured
# 2026-09-01 from data/processed/equity.parquet just before PATCH 3 (dedupe
# platforms -> physical stations, then decay once per station). Patch 3 is a
# granularity correction; magnitudes are expected to stay close.
FLAGGED_PATCH2 = {
    "North Toronto":       {"access": 0.2953, "gap": 0.4297},
    "Yonge-Doris":         {"access": 0.3387, "gap": 0.3843},
    "Church-Wellesley":    {"access": 0.4790, "gap": 0.3799},
    "North St.James Town": {"access": 0.4104, "gap": 0.4570},
}
# per-PLATFORM walkshed credit + access under patch 2, for the two neighbourhoods
# whose downtown platform pile-up motivated patch 3.
PATCH2_PLATFORM = {
    "University":     {"credit": 4320.2, "access": 0.5005},
    "Bay-Cloverhill": {"credit": 3649.5, "access": 0.6631},
}


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
        "subway_stop_count", "rapid_transit_stop_count",
        "neighbourhood_frequency", "neighbourhood_capacity_weighted_frequency",
        "nearby_rapid_transit_credit", "total_effective_frequency",
        "neighbourhood_stop_density",
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

    # ---- 5a. PATCH 3 (station-level walkshed): patch 2 vs patch 3 ----
    n_rt_eff = int(((merged["rapid_transit_stop_count"] > 0)
                    | (merged["nearby_rapid_transit_credit"] > 0)).sum())
    print("\n" + "-" * 78)
    print("  PATCH 3 -- dedupe rapid-transit PLATFORMS -> physical STATIONS before")
    print("  the 500 m walkshed decay (one interchange = one distance-decayed credit).")
    print("\n  patch 2 (per-platform) vs patch 3 (per-station), flagged neighbourhoods:")
    for nm, p2 in FLAGGED_PATCH2.items():
        r = merged[merged["AREA_NAME"] == nm].iloc[0]
        da = r["access_score"] - p2["access"]
        dg = r["equity_gap"] - p2["gap"]
        print(f"    {nm:<22} access {p2['access']:.3f} -> {r['access_score']:.3f} ({da:+.3f})"
              f"   equity_gap {p2['gap']:+.3f} -> {r['equity_gap']:+.3f} ({dg:+.3f})")
        print(f"      {'':<20}   own cap-wt freq {r['neighbourhood_capacity_weighted_frequency']:7.1f}"
              f"  + station walkshed credit {r['nearby_rapid_transit_credit']:7.1f}"
              f"  = total_effective {r['total_effective_frequency']:7.1f}")

    print("\n  the two downtown neighbourhoods that motivated patch 3"
          " (per-platform -> per-station):")
    for nm, pp in PATCH2_PLATFORM.items():
        r = merged[merged["AREA_NAME"] == nm].iloc[0]
        print(f"    {nm:<16} walkshed credit {pp['credit']:7.1f} -> {r['nearby_rapid_transit_credit']:7.1f}"
              f"   access {pp['access']:.3f} -> {r['access_score']:.3f}")

    print(f"\n  neighbourhoods with rapid-transit access (own stop OR walkshed credit): "
          f"{n_rt_eff} of 158")

    print("\n  NORTH TORONTO -- full trajectory:")
    nt = merged[merged["AREA_NAME"] == "North Toronto"].iloc[0]
    p0 = {"access": 0.264181, "gap": 0.460880}   # Phase 3 original (plain frequency)
    p1 = FLAGGED_PATCH1["North Toronto"]
    p2 = FLAGGED_PATCH2["North Toronto"]
    print(f"    Phase 3 original : access {p0['access']:.3f}   equity_gap {p0['gap']:+.3f}   (plain frequency)")
    print(f"    after patch 1    : access {p1['access']:.3f}   equity_gap {p1['gap']:+.3f}   "
          f"(capacity weighting; 0 own subway stops -> barely moved)")
    print(f"    after patch 2    : access {p2['access']:.3f}   equity_gap {p2['gap']:+.3f}   "
          f"(Fix A LRT x6 on its Line 5 platforms; Fix B per-platform walkshed)")
    print(f"    after patch 3    : access {nt['access_score']:.3f}   equity_gap {nt['equity_gap']:+.3f}   "
          f"(Eglinton Stn's 4 platforms -> 1 station point; credit "
          f"{nt['nearby_rapid_transit_credit']:.0f}, ~same total, decayed once)")

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
