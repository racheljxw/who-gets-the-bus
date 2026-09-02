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
       stop_count,
       neighbourhood_capacity_weighted_frequency = SUM(capacity_weighted_trips_per_hour),
       neighbourhood_frequency = SUM(trips_per_hour)   [kept for audit only],
       nearby_rapid_transit_credit = walkshed credit from Fix B (below),
       total_effective_frequency = neighbourhood_capacity_weighted_frequency
                                   + nearby_rapid_transit_credit,
       area_km2, neighbourhood_stop_density = stop_count / area_km2,
       freq_norm    = min-max(total_effective_frequency)   -> 0..1
       density_norm = min-max(neighbourhood_stop_density)  -> 0..1
       raw_access   = 0.4*freq_norm + 0.6*density_norm
       access_score = raw_access   (already 0..1: weighted sum of two 0..1 terms)

     CAPACITY-WEIGHTING PATCH (2026-09-01, Phase 3 review): freq_norm uses a
     *capacity-weighted* frequency (each stop_time scaled by vehicle capacity per
     mode -- subway x15, LRT x6, streetcar x2.5, bus x1; see
     scripts/02_stop_frequency.py and CLAUDE.md section 5). Before this, sparse
     rapid-transit neighbourhoods (North Toronto, Yonge-Doris) scored as
     "underserved" purely because a station is one physical stop while a bus
     corridor is many. Plain neighbourhood_frequency is still written for audit.

     PATCH 2 / FIX B (2026-09-01) -- rapid-transit walkshed credit. The
     point-in-polygon assignment in step 3 is unchanged and stays primary, but a
     station 20 m outside a boundary (Eglinton Station vs North Toronto) still
     serves that neighbourhood in reality. So, ADDITIONALLY, credit nearby
     rapid-transit service with linear distance decay over CREDIT_RADIUS_M (500
     m). One station can credit several nearby neighbourhoods -- intentional, it
     mirrors walkable access; it is NOT a join fix and does not move the stop's
     primary assignment. Every credit is printed with station + distance.

     PATCH 3 (2026-09-01) -- dedupe platforms into physical STATIONS before
     applying decay. GTFS lists each platform as its own stop, so patch 2
     credited a 4-platform interchange four times for one physical walk. Now:
     group rapid-transit platforms by GTFS parent_station (checked first; this
     feed has none populated) else by normalized stop_name (suffix stripped;
     grouping printed for audit). Per station: total_station_frequency = SUM of
     platform capacity_weighted_trips_per_hour; location = centroid of platform
     coords (EPSG:2952). Then for every polygon P not containing a station S's
     centroid, if dist(S, P) <= 500 m:
        credit(P) += total_station_frequency(S) * (1 - dist / 500)
     nearby_rapid_transit_credit = SUM of those; total_effective_frequency =
     neighbourhood_capacity_weighted_frequency (own stops) + that credit;
     freq_norm = min-max(total_effective_frequency).

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
import re
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
RESCUE_TOL_M = 25       # snap a just-outside stop to its nearest polygon within this many metres
CREDIT_RADIUS_M = 500.0  # Fix B: rapid-transit walkshed radius, linear decay to 0 at the edge


def minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if hi == lo:
        return pd.Series(0.0, index=s.index)
    return (s - lo) / (hi - lo)


# Fix B / patch 3: collapse platform stop_names to a physical-station name by
# stripping the direction / platform suffix. Handles the shapes actually seen in
# this feed (audited in 03's output):
#   "X Station - Northbound Platform"                 (subway, dashed)
#   "X Station - Northbound Platform Towards Finch"   (Union, trailing text)
#   "X Station - Subway Platform"                     (interchange subway side)
#   "X Station Eastbound Platform"                    (Line 5/6, no dash)
#   "X Station LRT Platform"                          (Line 5/6 LRT side)
_PLATFORM_SUFFIX_RE = re.compile(
    r"\s*[-–]?\s*(?:north|south|east|west)bound\s+platform.*$"
    r"|\s*[-–]?\s*subway\s+platform\s*$"
    r"|\s*[-–]?\s*lrt\s+platform\s*$"
    r"|\s*[-–]?\s*platform\s+[A-Za-z0-9]+\s*$",
    re.IGNORECASE,
)


def normalize_station_name(name: str) -> str:
    return _PLATFORM_SUFFIX_RE.sub("", str(name or "").strip()).strip()


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

    STOP_COLS = ["stop_id", "trips_per_hour", "capacity_weighted_trips_per_hour",
                 "has_subway_route", "has_rapid_transit", "geometry"]

    # ---- 3. sjoin intersects + dedup ----
    joined = gpd.sjoin(
        stops[STOP_COLS],
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
        far = stops.loc[unmatched_idx, STOP_COLS]
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
            [matched, near[["stop_id", "trips_per_hour", "capacity_weighted_trips_per_hour",
                            "has_subway_route", "has_rapid_transit",
                            "AREA_SHORT_CODE", "AREA_NAME", "geometry"]]],
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

    # ---- 3b. FIX B (patch 3): STATION-level rapid-transit walkshed credit ----
    # Does NOT touch step 3's point-in-polygon assignment. Patch 3 change: dedupe
    # platforms into physical stations FIRST, so a 4-platform interchange is one
    # distance-decayed contribution, not four. Then for each station S and each
    # polygon P whose interior does not contain S's centroid, if the centroid is
    # <= CREDIT_RADIUS_M from P: credit(P) += total_station_frequency(S) *
    # (1 - dist/CREDIT_RADIUS_M). Distances in EPSG:2952 metres.
    rt = stops.loc[
        stops["has_rapid_transit"] & (stops["capacity_weighted_trips_per_hour"] > 0),
        ["stop_id", "stop_name", "parent_station",
         "capacity_weighted_trips_per_hour", "geometry"],
    ].copy()

    n_parent = int(rt["parent_station"].notna().sum())
    print(f"\n  Fix B (station-level walkshed, {CREDIT_RADIUS_M:.0f} m linear decay):")
    print(f"    {len(rt)} rapid-transit platforms with service; "
          f"GTFS parent_station populated on {n_parent} / {len(rt)}")
    if n_parent == len(rt):
        rt["station_key"] = rt["parent_station"].astype(str)
        rt["station_name"] = rt["station_key"]
        print("    -> grouping platforms by parent_station")
    else:
        rt["station_name"] = rt["stop_name"].map(normalize_station_name)
        rt["station_key"] = rt["station_name"]
        if n_parent:
            has_p = rt["parent_station"].notna()
            rt.loc[has_p, "station_key"] = rt.loc[has_p, "parent_station"].astype(str)
        print(f"    -> parent_station unusable; grouping {len(rt)} platforms by "
              f"normalized stop_name")

    grp = rt.groupby("station_key")
    stn = pd.DataFrame({
        "station_name": grp["station_name"].first(),
        "total_station_frequency": grp["capacity_weighted_trips_per_hour"].sum(),
        "n_platforms": grp.size(),
        "x": rt.geometry.x.groupby(rt["station_key"]).mean(),   # centroid of
        "y": rt.geometry.y.groupby(rt["station_key"]).mean(),   # platform coords
    }).reset_index()
    stn = gpd.GeoDataFrame(
        stn, geometry=gpd.points_from_xy(stn["x"], stn["y"]), crs=nbhd.crs
    )
    # intra-station platform spread -- guards against a bad name merge
    _cx = rt.geometry.x.groupby(rt["station_key"]).transform("mean")
    _cy = rt.geometry.y.groupby(rt["station_key"]).transform("mean")
    rt["_spread_m"] = ((rt.geometry.x - _cx) ** 2 + (rt.geometry.y - _cy) ** 2) ** 0.5
    spread = rt.groupby("station_key")["_spread_m"].max()
    stn = stn.merge(spread.rename("platform_spread_m"), on="station_key")
    print(f"    {len(rt)} platforms -> {len(stn)} physical stations "
          f"(max {int(stn['n_platforms'].max())} platforms/station; "
          f"largest intra-station platform spread {stn['platform_spread_m'].max():.0f} m)")

    print("\n  STATION GROUPING AUDIT (station -> platform stop_ids absorbed):")
    for _, s in stn.sort_values("station_name").iterrows():
        ids = sorted(rt.loc[rt["station_key"] == s["station_key"], "stop_id"].tolist())
        flag = "  <-- CHECK spread" if s["platform_spread_m"] > 400 else ""
        print(f"    {s['station_name']:<42} n={int(s['n_platforms'])}  "
              f"freq={s['total_station_frequency']:7.1f}  ids={ids}{flag}")

    stn_buf = stn.copy()
    stn_buf["geometry"] = stn_buf.geometry.buffer(CREDIT_RADIUS_M)
    cand = gpd.sjoin(
        stn_buf, nbhd[["AREA_SHORT_CODE", "AREA_NAME", "geometry"]],
        how="inner", predicate="intersects",
    ).drop(columns="index_right")

    poly_geom = nbhd.set_index("AREA_SHORT_CODE").geometry
    pt_geom = stn.set_index("station_key").geometry
    cand["dist_m"] = [
        poly_geom.loc[code].distance(pt_geom.loc[key])
        for key, code in zip(cand["station_key"], cand["AREA_SHORT_CODE"])
    ]
    # dist == 0 -> polygon contains the station centroid; not "nearby", skip.
    cred = cand[(cand["dist_m"] > 0) & (cand["dist_m"] <= CREDIT_RADIUS_M)].copy()
    cred["credit"] = cred["total_station_frequency"] * (
        1.0 - cred["dist_m"] / CREDIT_RADIUS_M
    )
    credit_by_nbhd = (
        cred.groupby("AREA_SHORT_CODE")["credit"].sum()
        .rename("nearby_rapid_transit_credit")
    )
    print(f"\n    {len(cred)} (station, neighbourhood) credit pairs -> "
          f"{cred['AREA_SHORT_CODE'].nunique()} neighbourhoods get some credit")

    # ---- 4. aggregate per neighbourhood (all 158) ----
    agg = (
        matched.groupby("AREA_SHORT_CODE")
        .agg(stop_count=("stop_id", "size"),
             neighbourhood_frequency=("trips_per_hour", "sum"),
             neighbourhood_capacity_weighted_frequency=(
                 "capacity_weighted_trips_per_hour", "sum"),
             subway_stop_count=("has_subway_route", "sum"),
             rapid_transit_stop_count=("has_rapid_transit", "sum"))
        .reset_index()
    )

    acc = nbhd[["AREA_SHORT_CODE", "AREA_NAME", "area_km2"]].merge(
        agg, on="AREA_SHORT_CODE", how="left"
    )
    acc["stop_count"] = acc["stop_count"].fillna(0).astype(int)
    acc["neighbourhood_frequency"] = acc["neighbourhood_frequency"].fillna(0.0)
    acc["neighbourhood_capacity_weighted_frequency"] = (
        acc["neighbourhood_capacity_weighted_frequency"].fillna(0.0)
    )
    acc["subway_stop_count"] = acc["subway_stop_count"].fillna(0).astype(int)
    acc["rapid_transit_stop_count"] = acc["rapid_transit_stop_count"].fillna(0).astype(int)
    acc = acc.merge(credit_by_nbhd, on="AREA_SHORT_CODE", how="left")
    acc["nearby_rapid_transit_credit"] = acc["nearby_rapid_transit_credit"].fillna(0.0)
    acc["total_effective_frequency"] = (
        acc["neighbourhood_capacity_weighted_frequency"]
        + acc["nearby_rapid_transit_credit"]
    )
    acc["neighbourhood_stop_density"] = acc["stop_count"] / acc["area_km2"]

    # Normalize each term to 0..1 FIRST, then combine (revised formula 2026-09-01).
    # freq_norm uses total_effective_frequency: own capacity-weighted stops (Fix A
    # weights) + the Fix B walkshed credit. Plain neighbourhood_frequency and the
    # own-stops-only capacity-weighted frequency are kept for audit.
    acc["freq_norm"] = minmax(acc["total_effective_frequency"])
    acc["density_norm"] = minmax(acc["neighbourhood_stop_density"])
    acc["raw_access"] = (
        FREQ_WEIGHT * acc["freq_norm"] + DENSITY_WEIGHT * acc["density_norm"]
    )
    # raw_access is already in 0..1 (weighted sum of two 0..1 terms, weights sum
    # to 1), so access_score == raw_access -- no second normalization.
    acc["access_score"] = acc["raw_access"]

    acc = acc[[
        "AREA_SHORT_CODE", "AREA_NAME", "stop_count", "subway_stop_count",
        "rapid_transit_stop_count", "neighbourhood_frequency",
        "neighbourhood_capacity_weighted_frequency", "nearby_rapid_transit_credit",
        "total_effective_frequency", "area_km2", "neighbourhood_stop_density",
        "freq_norm", "density_norm", "raw_access", "access_score",
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

    print("\n  formula: raw_access = 0.4*freq_norm + 0.6*density_norm  "
          "(each term min-max'd to 0..1 first); access_score = raw_access.")
    print("  freq_norm = minmax(total_effective_frequency), where")
    print("    total_effective_frequency = neighbourhood_capacity_weighted_frequency "
          "(own stops, Fix A weights: subway x15 / LRT x6 / streetcar x2.5 / bus x1)")
    print(f"                              + nearby_rapid_transit_credit "
          f"(Fix B: <= {CREDIT_RADIUS_M:.0f} m walkshed, linear decay)")
    print("  raw term ranges (pre-normalization):")
    print(f"    neighbourhood_frequency (plain, audit)     : "
          f"{acc['neighbourhood_frequency'].min():.1f} "
          f".. {acc['neighbourhood_frequency'].max():.1f}  "
          f"(median {acc['neighbourhood_frequency'].median():.1f})")
    cwf = acc["neighbourhood_capacity_weighted_frequency"]
    print(f"    capacity_weighted_freq (own stops, audit)  : {cwf.min():.1f} "
          f".. {cwf.max():.1f}  (median {cwf.median():.1f})")
    tef = acc["total_effective_frequency"]
    print(f"    total_effective_frequency  <- feeds freq_norm: {tef.min():.1f} "
          f".. {tef.max():.1f}  (median {tef.median():.1f})")
    print(f"    neighbourhood_stop_density                 : {acc['neighbourhood_stop_density'].min():.1f} "
          f".. {acc['neighbourhood_stop_density'].max():.1f}  "
          f"(median {acc['neighbourhood_stop_density'].median():.1f})  [stops/km^2]")

    # ---- Fix B audit trail: every neighbourhood with credit > 0 ----
    got = acc[acc["nearby_rapid_transit_credit"] > 0].sort_values(
        "nearby_rapid_transit_credit", ascending=False
    )
    print(f"\n  FIX B AUDIT -- {len(got)} neighbourhoods receive walkshed credit "
          f"(why a no-own-stop score can still move); crediting STATIONS not platforms:")
    for _, r in got.iterrows():
        code = r["AREA_SHORT_CODE"]
        rows = cred[cred["AREA_SHORT_CODE"] == code].sort_values("credit", ascending=False)
        print(f"    {r['AREA_NAME']:<34} +{r['nearby_rapid_transit_credit']:7.1f}  "
              f"(own cap-wt freq {r['neighbourhood_capacity_weighted_frequency']:7.1f} "
              f"-> total {r['total_effective_frequency']:7.1f})")
        for _, c in rows.iterrows():
            print(f"        {c['dist_m']:5.0f} m  x(1-d/{CREDIT_RADIUS_M:.0f})="
                  f"{1 - c['dist_m']/CREDIT_RADIUS_M:.2f}  +{c['credit']:6.1f}  "
                  f"<- {c['station_name']} ({int(c['n_platforms'])} plat, "
                  f"freq {c['total_station_frequency']:.0f})")

    def _row(r):
        return (f"    {r['access_score']:.3f}  {r['AREA_NAME']:<38} "
                f"stops={r['stop_count']:3d}  "
                f"teff={r['total_effective_frequency']:8.1f} "
                f"(fn={r['freq_norm']:.3f})  dens={r['neighbourhood_stop_density']:5.1f} "
                f"(dn={r['density_norm']:.3f})")

    print("\n  TOP 5 access_score:")
    for _, r in acc.head(5).iterrows():
        print(_row(r))

    print("\n  BOTTOM 5 access_score:")
    for _, r in acc.tail(5).iterrows():
        print(_row(r))

    n_subway_nbhd = int((acc["subway_stop_count"] > 0).sum())
    n_rt_own = int((acc["rapid_transit_stop_count"] > 0).sum())
    n_rt_eff = int(((acc["rapid_transit_stop_count"] > 0)
                    | (acc["nearby_rapid_transit_credit"] > 0)).sum())
    print(f"\n  neighbourhoods with >= 1 rapid-transit stop of their own: {n_rt_own} of 158")
    print(f"  neighbourhoods with rapid-transit access (own stop OR walkshed credit): "
          f"{n_rt_eff} of 158")

    print("\n  patch-3 (station-level walkshed) on the previously-flagged neighbourhoods:")
    for nm in ("North Toronto", "Yonge-Doris", "Church-Wellesley",
               "North St.James Town", "West Humber-Clairville"):
        m = acc[acc["AREA_NAME"] == nm]
        if len(m):
            r = m.iloc[0]
            rank = acc.index[acc["AREA_NAME"] == nm][0] + 1
            print(f"    {nm:<24} access_score={r['access_score']:.3f} (rank {rank}/158)  "
                  f"own_cwfreq={r['neighbourhood_capacity_weighted_frequency']:7.1f}  "
                  f"+credit={r['nearby_rapid_transit_credit']:6.1f}  "
                  f"fn={r['freq_norm']:.3f}")

    zero = acc[acc["stop_count"] == 0]
    print(f"\n  zero-stop neighbourhoods: {len(zero)}  "
          f"(rule still implemented: access stays in, not excluded -- just never triggered)")
    for _, r in zero.iterrows():
        print(f"    {r['AREA_NAME']:<38}  raw_access={r['raw_access']:.4f}  "
              f"access_score={r['access_score']:.4f}")

    print(f"\n  overall access_score range: {acc['access_score'].min():.4f} "
          f".. {acc['access_score'].max():.4f}")

    print("\nPhase 1b done. Next: Phase 2 demographics -> need_score (needs confirmation).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
