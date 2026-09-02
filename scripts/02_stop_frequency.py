"""
Phase 1a -- GTFS -> per-stop weekday morning trip frequency.

Inputs:  data/raw/opendata_ttc_schedules.zip   (TTC GTFS feed)
Outputs: data/processed/stop_frequency.parquet
         one row per stop in stops.txt -- stop_id, stop_name, parent_station,
         stop_lat, stop_lon, trip_count, trips_per_hour,
         capacity_weighted_trip_count, capacity_weighted_trips_per_hour,
         subway_trip_count, has_subway_route, rapid_transit_trip_count,
         has_rapid_transit. Zero-service stops are kept with 0s.

Pipeline:
  1. Pick a representative *normal weekday* date inside the feed's valid range,
     skipping Labour Day (calendar_dates removes service_id 1 that day). The
     actual filter is on service_id, not the date; the date only justifies the
     choice and is confirmed active via partridge.
  2. Load the feed through partridge with a view that keeps ONLY service_id == 1
     trips. Partridge propagates that filter to stop_times automatically.
  3. Filter stop_times to rows whose arrival_time OR departure_time falls in the
     07:00:00-09:00:00 window. GTFS clock strings can exceed 24:00:00 for
     after-midnight trips, so times are compared as *seconds since midnight*
     (h may be >= 24), never as wall-clock.
  4. trips_per_hour(stop) = (qualifying stop_times at that stop) / 2, lat/lon
     joined from stops.txt.
  5. Write the parquet.

Why capacity weighting (deviation from a plain trip count -- see CLAUDE.md §5):
  A plain stop_time count treats a 6-car subway train and a single bus as equal.
  Combined with the stop-density term in 03_access_score.py (which favours
  closely-spaced bus stops), that made rapid-transit neighbourhoods score as
  "underserved" purely for having few physical stops. So each qualifying
  stop_time is scaled by its route's approximate vehicle capacity, normalized to
  a standard bus = 1.0 (see CAPACITY_MULTIPLIER). Each stop_time is weighted by
  its OWN mode before summing; a multi-mode stop is not averaged.

Why route_type==0 is split into streetcar vs lrt:
  This feed codes BOTH the legacy streetcar network (Queen, King, Spadina, ...)
  AND the modern Flexity-Freedom light-rail lines ("Line 5 Eglinton",
  "Line 6 Finch West") as route_type==0, but they are very different vehicles.
  A route is "lrt" iff route_long_name begins "Line <digit>" (matched
  generically, and every route_type==0 route is printed for audit before the
  rule is applied -- nothing is hard-coded by id); all other route_type==0 is
  "streetcar". has_rapid_transit (subway OR lrt) drives the walkshed credit in
  03_access_score.py.

Usage:
    python scripts/02_stop_frequency.py
"""

from __future__ import annotations

import os
import re
import sys
import zipfile
from datetime import date, timedelta

import pandas as pd
import partridge as ptg

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

HERE = os.path.dirname(__file__)
RAW_ZIP = os.path.abspath(os.path.join(HERE, "..", "data", "raw", "opendata_ttc_schedules.zip"))
OUT_PARQUET = os.path.abspath(os.path.join(HERE, "..", "data", "processed", "stop_frequency.parquet"))

WEEKDAY_SERVICE_ID = "1"          # normal Mon-Fri service (Phase 0 finding)
LABOUR_DAY = date(2026, 9, 7)     # service_id 1 removed this day via calendar_dates.txt

PEAK_START_SEC = 7 * 3600         # 07:00:00 -> 25200
PEAK_END_SEC = 9 * 3600           # 09:00:00 -> 32400
WINDOW_HOURS = 2.0

# Approximate per-vehicle service capacity by mode, normalized to a standard
# bus = 1.0. These are order-of-magnitude service-capacity ratios (nominal
# in-service loads), NOT crush loads or measured ridership -- see CLAUDE.md §5:
#   bus       ~50 passengers  -- standard TTC bus, the baseline
#   streetcar ~130            -- one low-floor Flexity Outlook streetcar
#   lrt       ~295-300        -- Line 5/6 Flexity Freedom LRVs run COUPLED;
#                               intermediate between a streetcar and a subway
#   subway    ~800-900        -- 6-car T1 / Toronto Rocket trainset
# Unknown mode ("other") warns + falls back to 1.0 (see CAPACITY_FALLBACK).
CAPACITY_MULTIPLIER = {
    "bus": 1.0,
    "streetcar": 2.5,
    "lrt": 6.0,
    "subway": 15.0,
}
CAPACITY_FALLBACK = 1.0
RAPID_TRANSIT_MODES = {"subway", "lrt"}   # feeds the walkshed credit in 03
# route_type=0 route is modern LRT (not a legacy streetcar) iff its long name
# looks like "Line 5 Eglinton" / "Line 6 Finch West". Confirmed by printing every
# route_type=0 route in the feed before this is applied -- see main().
LRT_LONGNAME_RE = re.compile(r"^\s*Line\s+\d")


def classify_mode(route_type: int, route_long_name: str) -> str:
    if route_type == 1:
        return "subway"
    if route_type == 3:
        return "bus"
    if route_type == 0:
        return "lrt" if LRT_LONGNAME_RE.match(str(route_long_name or "")) else "streetcar"
    return "other"


def gtfs_time_to_seconds(series: pd.Series) -> pd.Series:
    """
    Return seconds since midnight (h may be >= 24 for after-midnight trips).

    partridge already parses GTFS time columns to float seconds-since-midnight,
    so a numeric column passes straight through. We still handle the raw
    'HH:MM:SS' string form defensively in case that ever changes.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series.astype("float64")
    parts = series.astype("string").str.strip().str.split(":", expand=True)
    h = pd.to_numeric(parts[0], errors="coerce")
    m = pd.to_numeric(parts[1], errors="coerce")
    sec = pd.to_numeric(parts[2], errors="coerce")
    return h * 3600 + m * 60 + sec


def count_raw_stop_times_rows(zip_path: str) -> int:
    """Cheap line count of stop_times.txt straight from the zip (minus header)."""
    with zipfile.ZipFile(zip_path) as zf, zf.open("stop_times.txt") as fh:
        n = sum(1 for _ in fh)
    return n - 1  # header


def pick_representative_date(zip_path: str) -> date:
    """First Tue/Wed/Thu in the feed range on which service_id 1 is active."""
    svc_by_date = ptg.read_service_ids_by_date(zip_path)
    active_dates = sorted(svc_by_date)
    feed_start, feed_end = active_dates[0], active_dates[-1]
    print(f"  feed service calendar spans {feed_start} .. {feed_end}")

    d = feed_start
    while d <= feed_end:
        # Tue=1, Wed=2, Thu=3 -> mid-week, safest "typical" weekday
        if d.weekday() in (1, 2, 3) and d != LABOUR_DAY:
            if WEEKDAY_SERVICE_ID in {str(x) for x in svc_by_date[d]}:
                return d
        d += timedelta(days=1)
    raise RuntimeError("no mid-week date with service_id 1 found in feed range")


def main() -> int:
    if not os.path.isfile(RAW_ZIP):
        print(f"missing {RAW_ZIP} -- run scripts/00_fetch_data.py --download")
        return 1

    print("=" * 78)
    print("Phase 1a: GTFS -> per-stop weekday 7-9am trip frequency")
    print("=" * 78)

    chosen = pick_representative_date(RAW_ZIP)
    print(f"\n  CHOSEN DATE: {chosen} ({chosen.strftime('%A')})")
    print(f"    - inside feed validity window")
    print(f"    - mid-week (Tue/Wed/Thu): representative of a 'normal' weekday, "
          f"avoids Mon/Fri edge behaviour")
    print(f"    - not 2026-09-07 (Labour Day; service_id 1 is removed then)")
    print(f"    - partridge confirms service_id {WEEKDAY_SERVICE_ID} is active on this date")

    # ---- load feed: ONLY service_id == 1 trips (propagates to stop_times) ----
    view = {"trips.txt": {"service_id": WEEKDAY_SERVICE_ID}}
    feed = ptg.load_feed(RAW_ZIP, view=view)

    trips = feed.trips
    stop_times = feed.stop_times
    stops = feed.stops
    routes = feed.routes
    print(f"\n  service_id={WEEKDAY_SERVICE_ID}: {len(trips):,} trips")

    raw_rows = count_raw_stop_times_rows(RAW_ZIP)
    rows_service1 = len(stop_times)
    print(f"\n  stop_times rows -- full feed file : {raw_rows:,}")
    print(f"  stop_times rows -- service_id=1    : {rows_service1:,}")

    # ---- parse both time columns as seconds since midnight ----
    arr = gtfs_time_to_seconds(stop_times["arrival_time"])
    dep = gtfs_time_to_seconds(stop_times["departure_time"])

    over_24h = int(((arr >= 86400) | (dep >= 86400)).sum())
    print(f"  stop_times (service_id=1) with a 24:00:00+ time: {over_24h:,} "
          f"(expected 0 inside a 7-9am window; validated, not assumed)")

    in_window = (
        arr.between(PEAK_START_SEC, PEAK_END_SEC, inclusive="both")
        | dep.between(PEAK_START_SEC, PEAK_END_SEC, inclusive="both")
    ).fillna(False)

    peak = stop_times.loc[in_window].copy()
    print(f"  stop_times rows -- after 7-9am filter: {len(peak):,}  "
          f"({len(peak) / rows_service1:.1%} of service_id=1 rows)")

    # ---- route_type / mode per qualifying stop_time (trips.txt -> routes.txt) ----
    route_type = pd.to_numeric(routes["route_type"], errors="coerce").astype("Int64")
    routes_slim = routes[["route_id", "route_short_name", "route_long_name"]].assign(
        route_type=route_type
    )

    # Print EVERY route_type=0 route so the streetcar/LRT split is confirmed from
    # the feed, not guessed.
    rt0 = routes_slim[routes_slim["route_type"] == 0]
    print(f"\n  route_type=0 routes in this feed ({len(rt0)}): "
          f"legacy streetcar vs modern LRT --")
    for _, r in rt0.sort_values("route_id").iterrows():
        mode = classify_mode(0, r["route_long_name"])
        print(f"    route_id {str(r['route_id']):<5} short={str(r['route_short_name']):<6} "
              f"long={str(r['route_long_name'])!r:<26} -> {mode}")

    routes_slim["mode"] = [
        classify_mode(int(rt) if pd.notna(rt) else -1, ln)
        for rt, ln in zip(routes_slim["route_type"], routes_slim["route_long_name"])
    ]
    trip_rt = trips[["trip_id", "route_id"]].merge(
        routes_slim[["route_id", "route_type", "mode"]], on="route_id", how="left"
    )
    assert trip_rt["route_type"].notna().all(), (
        "some service_id=1 trips have no route_type after joining routes.txt"
    )
    present = sorted(int(x) for x in trip_rt["route_type"].unique())
    print(f"\n  route_type values present among service_id=1 trips: {present}")

    peak = peak.merge(trip_rt[["trip_id", "mode"]], on="trip_id", how="left")
    assert peak["mode"].notna().all(), "some peak stop_times did not match a trip"

    unknown = sorted(set(peak["mode"].unique()) - set(CAPACITY_MULTIPLIER))
    if unknown:
        for m in unknown:
            n = int((peak["mode"] == m).sum())
            print(f"  WARNING: mode {m!r} not in CAPACITY_MULTIPLIER -- "
                  f"{n:,} peak stop_times default to {CAPACITY_FALLBACK} (bus-equivalent)")
    peak["capacity_multiplier"] = (
        peak["mode"].map(CAPACITY_MULTIPLIER).fillna(CAPACITY_FALLBACK)
    )
    print("\n  capacity weighting applied per stop_time (by mode):")
    for m in ["bus", "streetcar", "lrt", "subway", "other"]:
        n = int((peak["mode"] == m).sum())
        if n:
            print(f"    {m:<10} x{CAPACITY_MULTIPLIER.get(m, CAPACITY_FALLBACK):<4}  "
                  f"{n:,} peak stop_times")

    # ---- per-stop counts -> trips_per_hour (plain + capacity-weighted) ----
    peak["_is_subway"] = peak["mode"] == "subway"
    peak["_is_rapid"] = peak["mode"].isin(RAPID_TRANSIT_MODES)
    counts = (
        peak.groupby("stop_id")
        .agg(trip_count=("trip_id", "size"),
             capacity_weighted_trip_count=("capacity_multiplier", "sum"),
             subway_trip_count=("_is_subway", "sum"),
             rapid_transit_trip_count=("_is_rapid", "sum"))
        .reset_index()
    )
    counts["has_subway_route"] = counts["subway_trip_count"] > 0
    counts["has_rapid_transit"] = counts["rapid_transit_trip_count"] > 0
    counts["trips_per_hour"] = counts["trip_count"] / WINDOW_HOURS
    counts["capacity_weighted_trips_per_hour"] = (
        counts["capacity_weighted_trip_count"] / WINDOW_HOURS
    )

    # Left join from every stop in stops.txt, so zero-service stops are retained
    # as 0. stop_name + parent_station are carried through for the platform ->
    # station dedup that 03_access_score.py does for the walkshed credit.
    stop_cols = ["stop_id", "stop_lat", "stop_lon"]
    for extra in ("stop_name", "parent_station"):
        if extra in stops.columns:
            stop_cols.insert(1, extra)
    out = stops[stop_cols].merge(counts, on="stop_id", how="left")
    out["trip_count"] = out["trip_count"].fillna(0).astype(int)
    out["trips_per_hour"] = out["trips_per_hour"].fillna(0.0)
    out["capacity_weighted_trip_count"] = out["capacity_weighted_trip_count"].fillna(0.0)
    out["capacity_weighted_trips_per_hour"] = out["capacity_weighted_trips_per_hour"].fillna(0.0)
    out["subway_trip_count"] = out["subway_trip_count"].fillna(0).astype(int)
    out["has_subway_route"] = out["has_subway_route"].fillna(False).astype(bool)
    out["rapid_transit_trip_count"] = out["rapid_transit_trip_count"].fillna(0).astype(int)
    out["has_rapid_transit"] = out["has_rapid_transit"].fillna(False).astype(bool)
    out["stop_lat"] = pd.to_numeric(out["stop_lat"], errors="coerce")
    out["stop_lon"] = pd.to_numeric(out["stop_lon"], errors="coerce")

    os.makedirs(os.path.dirname(OUT_PARQUET), exist_ok=True)
    out.to_parquet(OUT_PARQUET, index=False)

    # ---- report ----
    served = out[out["trips_per_hour"] > 0]
    print("\n" + "-" * 78)
    print(f"  wrote {OUT_PARQUET}")
    print(f"  columns: {list(out.columns)}")
    print(f"  total stops in stops.txt              : {len(out):,}")
    print(f"  stops with trips_per_hour > 0         : {len(served):,}")
    print(f"  stops with trips_per_hour == 0        : {len(out) - len(served):,}")
    print(f"  stops served by a subway route (route_type=1)   : "
          f"{int(out['has_subway_route'].sum()):,}")
    print(f"  stops served by rapid transit (subway OR Line 5/6): "
          f"{int(out['has_rapid_transit'].sum()):,}")
    print(f"  trips_per_hour  max={served['trips_per_hour'].max():.1f}  "
          f"median={served['trips_per_hour'].median():.1f}  "
          f"mean={served['trips_per_hour'].mean():.2f}")
    cw = served["capacity_weighted_trips_per_hour"]
    print(f"  capacity_weighted_trips_per_hour  max={cw.max():.1f}  "
          f"median={cw.median():.1f}  mean={cw.mean():.2f}  "
          f"(vs plain -- diverges most at subway/streetcar stops)")

    print("\n  busiest 10 stops by plain trips_per_hour (7-9am):")
    top = out.sort_values("trips_per_hour", ascending=False).head(10)
    id_to_name = dict(zip(stops["stop_id"], stops.get("stop_name", pd.Series(dtype=str))))
    for _, r in top.iterrows():
        nm = id_to_name.get(r["stop_id"], "?")
        print(f"    stop {r['stop_id']:>7}  {r['trips_per_hour']:6.1f} tph  "
              f"(cap-wt {r['capacity_weighted_trips_per_hour']:7.1f})  {nm}")

    print("\n  busiest 10 stops by capacity_weighted_trips_per_hour (7-9am):")
    for _, r in out.sort_values("capacity_weighted_trips_per_hour", ascending=False).head(10).iterrows():
        nm = id_to_name.get(r["stop_id"], "?")
        print(f"    stop {r['stop_id']:>7}  cap-wt {r['capacity_weighted_trips_per_hour']:7.1f}  "
              f"(plain {r['trips_per_hour']:6.1f} tph)  {nm}")

    # sanity: named landmark stops (substring match on stop_name)
    print("\n  sanity -- known frequent Toronto locations (matching stops):")
    lookup = out.set_index("stop_id")[["trip_count", "trips_per_hour"]]
    for needle in ("Bloor", "Yonge", "Dufferin", "Finch", "Union"):
        hits = stops[stops["stop_name"].str.contains(needle, case=False, na=False)]
        if hits.empty:
            print(f"    '{needle}': no stop_name match")
            continue
        j = hits.merge(lookup, left_on="stop_id", right_index=True, how="left")
        j["trips_per_hour"] = j["trips_per_hour"].fillna(0.0)
        tot = j["trips_per_hour"].sum()
        busiest = j.sort_values("trips_per_hour", ascending=False).iloc[0]
        print(f"    '{needle}': {len(j)} stops, summed {tot:7.1f} tph; "
              f"busiest '{busiest['stop_name']}' @ {busiest['trips_per_hour']:.1f} tph")

    print("\nPhase 1a done. Next: Phase 1b spatial join stops -> neighbourhoods.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
