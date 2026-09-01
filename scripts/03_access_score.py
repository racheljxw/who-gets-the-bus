"""
Phase 1b -- spatial join stops -> neighbourhoods, then neighbourhood access_score.

Inputs:
  data/raw/neighbourhoods-4326.geojson        158 polygons, EPSG:4326
  data/processed/stop_frequency.parquet       per-stop 7-9am trips_per_hour (Phase 1a)

Steps (see CLAUDE.md section 5 + risk area 3):
  1. Load polygons, reproject to EPSG:2952 (MTM zone 10, metres) for area/density.
  2. Load stop frequency, build Point geoms from stop_lon/lat (EPSG:4326),
     reproject to EPSG:2952.
  3. sjoin(predicate='intersects') so boundary-line stops are not dropped.
     A stop exactly on a shared edge can match >1 polygon -> dedup by stop_id,
     keeping the match with the smallest AREA_SHORT_CODE (deterministic), so no
     stop is ever counted in two neighbourhoods. Report every deduped stop.
  4. Per neighbourhood (LEFT join from all 158 polygons; zero-stop kept):
       stop_count, neighbourhood_frequency = SUM(trips_per_hour),
       area_km2, neighbourhood_stop_density = stop_count / area_km2,
       freq_norm    = min-max(neighbourhood_frequency)     -> 0..1
       density_norm = min-max(neighbourhood_stop_density)  -> 0..1
       raw_access   = 0.4*freq_norm + 0.6*density_norm
       access_score = raw_access   (already 0..1: weighted sum of two 0..1 terms)

     DEVIATION FROM ORIGINAL SPEC (explicit human decision, 2026-09-01): the
     original spec was raw_access = 0.6*frequency + 0.4*stop_density then a
     single min-max. That let frequency dominate -- the raw terms differ ~37x in
     scale (frequency 39.5-2105.5 vs density 4.6-57.2), so density barely moved
     the score. We now normalize EACH term to 0..1 first, then combine, and flip
     the weights to 0.4 frequency / 0.6 density -- deliberately giving stop
     density more relative weight than the original spec. West Humber-Clairville's
     outlier compression is knowingly left as-is; not in scope for this change.
  5. Write data/processed/access.parquet
  6. Print summary: top/bottom 5 by access_score, zero-stop list + their scores,
     and the raw scale gap between the frequency and density terms.

Phase 2 (demographics -> need_score) is next and needs explicit confirmation.

Usage:
    python scripts/03_access_score.py
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
GEOJSON = os.path.abspath(os.path.join(HERE, "..", "data", "raw", "neighbourhoods-4326.geojson"))
STOP_FREQ = os.path.abspath(os.path.join(HERE, "..", "data", "processed", "stop_frequency.parquet"))
OUT_PARQUET = os.path.abspath(os.path.join(HERE, "..", "data", "processed", "access.parquet"))

PROJECTED_CRS = 2952   # MTM zone 10, metres -- for area + density
# Weights on the *normalized* terms. Flipped from the original spec's 0.6/0.4
# (freq/density) to 0.4/0.6 -- see module docstring, step 4. Explicit human
# decision 2026-09-01.
FREQ_WEIGHT = 0.4
DENSITY_WEIGHT = 0.6
RESCUE_TOL_M = 25      # snap a just-outside stop to its nearest polygon within this many metres


def minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if hi == lo:
        return pd.Series(0.0, index=s.index)
    return (s - lo) / (hi - lo)


def main() -> int:
    for p in (GEOJSON, STOP_FREQ):
        if not os.path.isfile(p):
            print(f"missing {p}")
            return 1

    print("=" * 78)
    print("Phase 1b: spatial join stops -> neighbourhoods -> access_score")
    print("=" * 78)

    # ---- 1. polygons -> projected CRS ----
    nbhd = gpd.read_file(GEOJSON)[["AREA_SHORT_CODE", "AREA_NAME", "geometry"]].copy()
    nbhd["AREA_SHORT_CODE"] = nbhd["AREA_SHORT_CODE"].astype(int)
    nbhd = nbhd.to_crs(PROJECTED_CRS)
    nbhd["area_km2"] = nbhd.geometry.area / 1e6
    print(f"  {len(nbhd)} neighbourhood polygons, reprojected EPSG:4326 -> EPSG:{PROJECTED_CRS}")
    print(f"  area_km2: min={nbhd['area_km2'].min():.3f}  max={nbhd['area_km2'].max():.3f}  "
          f"sum={nbhd['area_km2'].sum():.1f}")

    # ---- 2. stops -> points -> projected CRS ----
    sf = pd.read_parquet(STOP_FREQ)
    stops = gpd.GeoDataFrame(
        sf,
        geometry=gpd.points_from_xy(sf["stop_lon"], sf["stop_lat"]),
        crs=4326,
    ).to_crs(PROJECTED_CRS)
    print(f"\n  {len(stops)} stops loaded; "
          f"{(stops['trips_per_hour'] > 0).sum()} with trips_per_hour > 0")

    # ---- 3. sjoin intersects + dedup ----
    joined = gpd.sjoin(
        stops[["stop_id", "trips_per_hour", "geometry"]],
        nbhd[["AREA_SHORT_CODE", "AREA_NAME", "geometry"]],
        how="left",
        predicate="intersects",
    )

    unmatched_idx = joined[joined["AREA_SHORT_CODE"].isna()].index
    matched = joined.dropna(subset=["AREA_SHORT_CODE"]).copy()
    matched["AREA_SHORT_CODE"] = matched["AREA_SHORT_CODE"].astype(int)

    print(f"\n  stops with no intersecting polygon: {len(unmatched_idx)}")
    if len(unmatched_idx):
        # 'intersects' still misses points that sit a few metres outside the
        # polygon because of GPS jitter / boundary generalization -- exactly the
        # "boundary-line stop" case we don't want to silently drop. Rescue only
        # points within RESCUE_TOL_M of a polygon via nearest join; anything
        # further out is genuinely outside Toronto (TTC serves some Mississauga /
        # York Region stops) and is left unassigned.
        far = stops.loc[unmatched_idx, ["stop_id", "trips_per_hour", "geometry"]]
        near = gpd.sjoin_nearest(
            far, nbhd[["AREA_SHORT_CODE", "AREA_NAME", "geometry"]],
            how="left", max_distance=RESCUE_TOL_M, distance_col="_dist_m",
        )
        near = near.dropna(subset=["AREA_SHORT_CODE"]).copy()
        near["AREA_SHORT_CODE"] = near["AREA_SHORT_CODE"].astype(int)
        # nearest can tie two polygons for an on-edge point -> dedup downstream
        rescued_ids = set(near["stop_id"])
        still_out = far[~far["stop_id"].isin(rescued_ids)]
        print(f"    rescued by nearest-polygon (<= {RESCUE_TOL_M} m): {len(rescued_ids)} stops "
              f"({(near.drop_duplicates('stop_id')['trips_per_hour'] > 0).sum()} with service)")
        print(f"    still unassigned (genuinely outside Toronto): {len(still_out)} stops "
              f"({(still_out['trips_per_hour'] > 0).sum()} with service -- "
              f"their frequency is not credited to any neighbourhood)")
        matched = pd.concat(
            [matched, near[["stop_id", "trips_per_hour", "AREA_SHORT_CODE", "AREA_NAME", "geometry"]]],
            ignore_index=True,
        )
        unmatched = still_out
    else:
        unmatched = stops.iloc[0:0]

    dup_ids = matched["stop_id"].value_counts()
    dup_ids = dup_ids[dup_ids > 1].index.tolist()
    print(f"\n  stops on a shared boundary (matched >1 polygon), needing dedup: {len(dup_ids)}")
    if dup_ids:
        for sid in dup_ids:
            rows = matched[matched["stop_id"] == sid]
            codes = sorted(rows["AREA_SHORT_CODE"].tolist())
            names = " | ".join(
                rows.sort_values("AREA_SHORT_CODE")["AREA_NAME"].tolist()
            )
            kept = min(codes)
            tph = rows["trips_per_hour"].iloc[0]
            print(f"    stop {sid} ({tph:.1f} tph): codes {codes} -> kept {kept}")
            print(f"      {names}")

    # deterministic: keep smallest AREA_SHORT_CODE per stop
    matched = (
        matched.sort_values(["stop_id", "AREA_SHORT_CODE"])
        .drop_duplicates(subset="stop_id", keep="first")
    )
    assert matched["stop_id"].is_unique, "dedup failed -- stop_id still not unique"
    print(f"\n  {len(matched)} stop->neighbourhood assignments after dedup "
          f"(1 neighbourhood per stop)")

    # ---- 4. aggregate per neighbourhood (all 158) ----
    agg = (
        matched.groupby("AREA_SHORT_CODE")
        .agg(stop_count=("stop_id", "size"),
             neighbourhood_frequency=("trips_per_hour", "sum"))
        .reset_index()
    )

    acc = nbhd[["AREA_SHORT_CODE", "AREA_NAME", "area_km2"]].merge(
        agg, on="AREA_SHORT_CODE", how="left"
    )
    acc["stop_count"] = acc["stop_count"].fillna(0).astype(int)
    acc["neighbourhood_frequency"] = acc["neighbourhood_frequency"].fillna(0.0)
    acc["neighbourhood_stop_density"] = acc["stop_count"] / acc["area_km2"]

    # Normalize each term to 0..1 FIRST, then combine (revised formula 2026-09-01).
    acc["freq_norm"] = minmax(acc["neighbourhood_frequency"])
    acc["density_norm"] = minmax(acc["neighbourhood_stop_density"])
    acc["raw_access"] = (
        FREQ_WEIGHT * acc["freq_norm"] + DENSITY_WEIGHT * acc["density_norm"]
    )
    # raw_access is already in 0..1 (weighted sum of two 0..1 terms, weights sum
    # to 1), so access_score == raw_access -- no second normalization.
    acc["access_score"] = acc["raw_access"]

    acc = acc[[
        "AREA_SHORT_CODE", "AREA_NAME", "stop_count", "neighbourhood_frequency",
        "area_km2", "neighbourhood_stop_density", "freq_norm", "density_norm",
        "raw_access", "access_score",
    ]].sort_values("access_score", ascending=False).reset_index(drop=True)

    os.makedirs(os.path.dirname(OUT_PARQUET), exist_ok=True)
    acc.to_parquet(OUT_PARQUET, index=False)

    # ---- 5/6. report ----
    print("\n" + "-" * 78)
    print(f"  wrote {OUT_PARQUET}  ({len(acc)} rows)")
    assert len(acc) == 158, f"expected 158 neighbourhoods, got {len(acc)}"

    total_assigned = acc["stop_count"].sum()
    print(f"  stops assigned to a neighbourhood: {total_assigned} "
          f"(of {len(stops)} total; {len(unmatched)} outside the boundary set)")

    print("\n  REVISED formula: raw_access = 0.4*freq_norm + 0.6*density_norm  "
          "(each term min-max'd to 0..1 first); access_score = raw_access.")
    print("  raw term ranges (pre-normalization -- the scale gap this revision fixes):")
    print(f"    neighbourhood_frequency      : {acc['neighbourhood_frequency'].min():.1f} "
          f".. {acc['neighbourhood_frequency'].max():.1f}  "
          f"(median {acc['neighbourhood_frequency'].median():.1f})")
    print(f"    neighbourhood_stop_density   : {acc['neighbourhood_stop_density'].min():.1f} "
          f".. {acc['neighbourhood_stop_density'].max():.1f}  "
          f"(median {acc['neighbourhood_stop_density'].median():.1f})  [stops/km^2]")
    print(f"    -> ~{acc['neighbourhood_frequency'].max() / acc['neighbourhood_stop_density'].max():.0f}x "
          f"apart at the top end; now each is min-max'd to 0..1 BEFORE weighting, "
          f"so both terms have equal reach and the 0.4/0.6 weights actually bite.")

    def _row(r):
        return (f"    {r['access_score']:.3f}  {r['AREA_NAME']:<38} "
                f"stops={r['stop_count']:3d}  freq={r['neighbourhood_frequency']:7.1f} "
                f"(fn={r['freq_norm']:.3f})  dens={r['neighbourhood_stop_density']:5.1f} "
                f"(dn={r['density_norm']:.3f})")

    print("\n  TOP 5 access_score:")
    for _, r in acc.head(5).iterrows():
        print(_row(r))

    print("\n  BOTTOM 5 access_score:")
    for _, r in acc.tail(5).iterrows():
        print(_row(r))

    whc = acc[acc["AREA_NAME"].str.startswith("West Humber")]
    if len(whc):
        r = whc.iloc[0]
        rank = acc.index[acc["AREA_NAME"] == r["AREA_NAME"]][0] + 1
        print(f"\n  West Humber-Clairville outlier check (compression left as-is, per decision):")
        print(f"    was access_score 1.000 (rank 1) under old formula")
        print(f"    now access_score {r['access_score']:.3f} (rank {rank} of 158)  "
              f"freq_norm={r['freq_norm']:.3f}  density_norm={r['density_norm']:.3f}")
        print(f"    gap to #2 now {r['access_score'] - acc['access_score'].iloc[1]:+.3f} "
              f"(was +0.231)")

    zero = acc[acc["stop_count"] == 0]
    print(f"\n  zero-stop neighbourhoods: {len(zero)}  "
          f"(rule still implemented: access stays in, not excluded -- just never triggered)")
    for _, r in zero.iterrows():
        print(f"    {r['AREA_NAME']:<38}  raw_access={r['raw_access']:.4f}  "
              f"access_score={r['access_score']:.4f}")

    avon = acc[acc["AREA_NAME"] == "Avondale"]
    if len(avon):
        r = avon.iloc[0]
        rank = acc.index[acc["AREA_NAME"] == "Avondale"][0] + 1
        print(f"\n  Avondale (was the 0.000 floor under old formula):")
        print(f"    now access_score {r['access_score']:.4f} (rank {rank} of 158, "
              f"{'still near the bottom' if rank >= 154 else 'MOVED -- investigate'})")
    print(f"  overall access_score range: {acc['access_score'].min():.4f} "
          f".. {acc['access_score'].max():.4f}")

    print("\nPhase 1b done. Next: Phase 2 demographics -> need_score (needs confirmation).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
