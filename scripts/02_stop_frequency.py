"""
Phase 1a -- GTFS -> per-stop weekday morning trip frequency.

Pipeline:
  1. Pick a representative *normal weekday* date inside the feed's valid range
     (2026-09-06 .. 2026-10-31), skipping 2026-09-07 (Labour Day: calendar_dates
     removes service_id 1 that day). We confirm via partridge that service_id 1
     is actually active on the chosen date.
  2. Load the feed through partridge with a view that keeps ONLY service_id == 1
     trips. Partridge propagates that filter to stop_times automatically.
  3. Filter stop_times to rows whose arrival_time OR departure_time falls in the
     07:00:00-09:00:00 window. GTFS clock strings can exceed 24:00:00 for
     after-midnight trips, so we parse "HH:MM:SS" as *seconds since midnight*
     (h may be >= 24), never as wall-clock, then compare in seconds.
  4. trips_per_hour(stop) = (qualifying stop_times at that stop) / 2   [2-hour window]
     lat/lon joined from stops.txt.
  4b. CAPACITY WEIGHTING (2026-09-01 decision -- see CLAUDE.md section 5).
     Plain trip counts treat a 6-car subway train and a single bus as equal.
     They aren't: a subway station serves far more passenger capacity per
     vehicle, so sparse-but-high-capacity subway neighbourhoods (North Toronto,
     Yonge-Doris) were scoring as "underserved" purely because they have few
     physical stops. Fix: weight each qualifying stop_time by its route's
     vehicle capacity, via GTFS route_type (trips.txt -> routes.txt):

       capacity_multiplier = {
         3  bus        : 1.0    # ~50-passenger standard TTC bus -- baseline
         0  streetcar  : 2.5    # ~130-passenger low-floor Flexity streetcar
         1  subway     : 15.0   # ~800-900-passenger 6-car T1 / Toronto Rocket train
       }

     These are explicit, approximate, sourced-but-not-precise assumptions
     (nominal service capacities, not crush loads). Any other route_type gets a
     printed warning and falls back to 1.0 (bus-equivalent) rather than a silent
     guess. Each stop_time is weighted individually by its own route_type before
     summing -- a stop served by multiple modes is NOT averaged.

       capacity_weighted_trips_per_hour(stop) =
         SUM(capacity_multiplier[route_type] for each qualifying stop_time) / 2

  5. Write data/processed/stop_frequency.parquet
     columns: stop_id, stop_lat, stop_lon, trip_count, trips_per_hour,
              capacity_weighted_trip_count, capacity_weighted_trips_per_hour

Phase 1b (spatial join stops -> neighbourhoods) is deliberately NOT done here.

Usage:
    python scripts/02_stop_frequency.py
"""

from __future__ import annotations

import os
import sys
import zipfile
from datetime import date, timedelta

import numpy as np
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

# Approximate per-vehicle service capacity by GTFS route_type, normalized to a
# standard bus = 1.0. Sourced-but-not-precise (2026-09-01 decision, CLAUDE.md
# section 5): ~50-pax bus, ~130-pax Flexity streetcar, ~800-900-pax 6-car subway
# train. Unknown route_types warn + fall back to 1.0 (see CAPACITY_FALLBACK).
CAPACITY_MULTIPLIER = {
    3: 1.0,    # bus
    0: 2.5,    # streetcar / tram
    1: 15.0,   # subway / metro
}
CAPACITY_FALLBACK = 1.0
ROUTE_TYPE_NAME = {0: "streetcar", 1: "subway", 3: "bus"}


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

    # ---- route_type per qualifying stop_time (trips.txt -> routes.txt) ----
    route_type = pd.to_numeric(routes["route_type"], errors="coerce").astype("Int64")
    trip_rt = trips[["trip_id", "route_id"]].merge(
        routes[["route_id"]].assign(route_type=route_type), on="route_id", how="left"
    )
    assert trip_rt["route_type"].notna().all(), (
        "some service_id=1 trips have no route_type after joining routes.txt"
    )
    present = sorted(int(x) for x in trip_rt["route_type"].unique())
    present_labels = ", ".join(
        "{} ({})".format(t, ROUTE_TYPE_NAME.get(t, "?")) for t in present
    )
    print(f"\n  route_type values present among service_id=1 trips: {present_labels}")

    peak = peak.merge(trip_rt[["trip_id", "route_type"]], on="trip_id", how="left")
    assert peak["route_type"].notna().all(), "some peak stop_times did not match a trip"
    peak["route_type"] = peak["route_type"].astype(int)

    unknown = sorted(set(peak["route_type"].unique()) - set(CAPACITY_MULTIPLIER))
    if unknown:
        for t in unknown:
            n = int((peak["route_type"] == t).sum())
            print(f"  WARNING: route_type {t} not in CAPACITY_MULTIPLIER -- "
                  f"{n:,} peak stop_times default to {CAPACITY_FALLBACK} (bus-equivalent)")
    peak["capacity_multiplier"] = (
        peak["route_type"].map(CAPACITY_MULTIPLIER).fillna(CAPACITY_FALLBACK)
    )
    print("  capacity weighting applied per stop_time:")
    for t in present:
        mult = CAPACITY_MULTIPLIER.get(t, CAPACITY_FALLBACK)
        n = int((peak["route_type"] == t).sum())
        print(f"    route_type {t} ({ROUTE_TYPE_NAME.get(t, '?'):<9}) x{mult:<4}  "
              f"{n:,} peak stop_times")

    # ---- per-stop counts -> trips_per_hour (plain + capacity-weighted) ----
    peak["_is_subway"] = peak["route_type"] == 1
    counts = (
        peak.groupby("stop_id")
        .agg(trip_count=("trip_id", "size"),
             capacity_weighted_trip_count=("capacity_multiplier", "sum"),
             subway_trip_count=("_is_subway", "sum"))
        .reset_index()
    )
    counts["has_subway_route"] = counts["subway_trip_count"] > 0
    counts["trips_per_hour"] = counts["trip_count"] / WINDOW_HOURS
    counts["capacity_weighted_trips_per_hour"] = (
        counts["capacity_weighted_trip_count"] / WINDOW_HOURS
    )

    # every stop in stops.txt, so zero-service stops are retained as 0
    out = stops[["stop_id", "stop_lat", "stop_lon"]].merge(counts, on="stop_id", how="left")
    out["trip_count"] = out["trip_count"].fillna(0).astype(int)
    out["trips_per_hour"] = out["trips_per_hour"].fillna(0.0)
    out["capacity_weighted_trip_count"] = out["capacity_weighted_trip_count"].fillna(0.0)
    out["capacity_weighted_trips_per_hour"] = out["capacity_weighted_trips_per_hour"].fillna(0.0)
    out["subway_trip_count"] = out["subway_trip_count"].fillna(0).astype(int)
    out["has_subway_route"] = out["has_subway_route"].fillna(False).astype(bool)
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
    print(f"  stops served by a subway route (route_type=1): "
          f"{int(out['has_subway_route'].sum()):,}")
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
    named = stops.copy()
    named["arr"] = np.nan
    lookup = out.set_index("stop_id")[["trip_count", "trips_per_hour"]]
    for needle in ("Bloor", "Yonge", "Dufferin", "Finch", "Union"):
        hits = named[named["stop_name"].str.contains(needle, case=False, na=False)]
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
