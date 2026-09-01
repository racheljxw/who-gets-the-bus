"""
Phase 0 -- Step 4: inspect + validate the raw downloads before any processing.

NO heavy processing happens here -- stop_times.txt is deliberately left untouched
(that is Phase 1). This script only:

  1. GTFS: open data/raw/*.zip, list members + uncompressed sizes, and print
     EVERY row of calendar.txt (and calendar_dates.txt) plus the trips-per-
     service_id counts, so we can pick the normal weekday service_id for the
     Phase 1 7-9am filter.

  2. Boundaries vs profiles: load the neighbourhoods GeoJSON and the 2021
     158-model profiles XLSX, print their id/name columns side by side, and
     test whether the join key lines up (Toronto has renumbered neighbourhood
     ids historically -- see CLAUDE.md section 6).

  3. Sanity: row counts + sample rows from each source.

Usage:
    python scripts/01_validate.py

Structure confirmed during Phase 0 (this script asserts these still hold and
prints loudly if they don't):

  boundaries  neighbourhoods-4326.geojson
      158 features, EPSG:4326. Join key = AREA_SHORT_CODE (int, values 1..174
      with gaps -- NOT 1..158 sequential). Name = AREA_NAME.

  profiles    nbhd_2021_census_profile_full_158model.xlsx
      sheet 'hd2021_census_profile', WIDE layout: 2604 indicator rows x 159
      cols. Col 0 = indicator label; cols 1..158 = neighbourhoods. Row 0 =
      'Neighbourhood Name', row 1 = 'Neighbourhood Number' (matches
      AREA_SHORT_CODE), row 2 = 'TSNS 2020 Designation'.

  => The join is AREA_SHORT_CODE == 'Neighbourhood Number'. During Phase 0 all
     158 codes matched with identical (punctuation-normalized) names. Names
     alone do NOT match exactly (7 differ only by spacing/punctuation, e.g.
     "Cabbagetown-South St.James Town" vs "... St. James Town"), so join on the
     NUMBER, carry AREA_NAME as the display label.
"""

from __future__ import annotations

import glob
import os
import re
import sys
import zipfile

import pandas as pd

# Windows consoles default to cp1252; force UTF-8 so neighbourhood names with
# accented characters (and any stray glyphs) print instead of crashing.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001 - non-fatal; older/edge stdio
    pass

RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))

PROFILE_SHEET = "hd2021_census_profile"
PROFILE_NAME_ROW = "Neighbourhood Name"
PROFILE_NUM_ROW = "Neighbourhood Number"


def _norm(s: str) -> str:
    """Punctuation/space-insensitive name key for comparison only."""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _pick_one(pattern: str, label: str) -> str | None:
    matches = sorted(glob.glob(os.path.join(RAW_DIR, pattern)))
    if not matches:
        print(f"  !! no {label} found matching {pattern} in {RAW_DIR}")
        return None
    if len(matches) > 1:
        print(f"  !! multiple {label} files; using the first:\n     " + "\n     ".join(matches))
    return matches[0]


# --------------------------------------------------------------------------- #
# 1. GTFS calendar inspection
# --------------------------------------------------------------------------- #
def inspect_gtfs() -> None:
    print("=" * 78)
    print("1. GTFS  (data/raw/*.zip)")
    zpath = _pick_one("*.zip", "GTFS zip")
    if not zpath:
        return
    print(f"  zip: {zpath}\n")
    with zipfile.ZipFile(zpath) as zf:
        members = zf.namelist()
        print(f"  {len(members)} members:")
        for m in sorted(members):
            info = zf.getinfo(m)
            flag = "   <-- ~4.26M rows / ~400MB: DO NOT load in full (Phase 1 filters it)" \
                if m == "stop_times.txt" else ""
            print(f"    {m:22s} {info.file_size/1e6:9.2f} MB uncompressed{flag}")

        for fname in ("calendar.txt", "calendar_dates.txt"):
            if fname not in members:
                print(f"\n  ({fname} not present)")
                continue
            with zf.open(fname) as fh:
                df = pd.read_csv(fh, dtype=str)
            print(f"\n  ---- {fname}: {len(df)} rows ----")
            with pd.option_context("display.max_rows", None, "display.max_columns", None,
                                   "display.width", 200):
                print(df.to_string(index=False))

        if "trips.txt" in members:
            with zf.open("trips.txt") as fh:
                trips = pd.read_csv(fh, dtype=str)
            print(f"\n  ---- trips.txt: {len(trips)} rows ----")
            print("  trips per service_id:")
            print(trips["service_id"].value_counts().to_string())
            if "calendar.txt" in members:
                print("\n  -> Pick the weekday service_id: monday..friday = 1 in calendar.txt,")
                print("     typically also the one with the most trips above.")


# --------------------------------------------------------------------------- #
# 2 + 3. Boundaries vs profiles
# --------------------------------------------------------------------------- #
def load_boundaries():
    path = _pick_one("*.geojson", "boundaries GeoJSON")
    if not path:
        return None
    import geopandas as gpd
    gdf = gpd.read_file(path)
    print(f"\n  boundaries: {os.path.basename(path)}")
    print(f"    rows: {len(gdf)}  (expect 158)   crs: {gdf.crs}")
    print(f"    columns: {[c for c in gdf.columns if c != 'geometry']}")
    if "AREA_SHORT_CODE" not in gdf.columns or "AREA_NAME" not in gdf.columns:
        print("    !! expected AREA_SHORT_CODE / AREA_NAME not found -- structure changed.")
        return gdf
    codes = sorted(int(x) for x in gdf["AREA_SHORT_CODE"])
    print(f"    AREA_SHORT_CODE: n={len(codes)} min={codes[0]} max={codes[-1]} "
          f"(sequential 1..158? {codes == list(range(1, 159))})")
    print("    sample rows [AREA_SHORT_CODE, AREA_LONG_CODE, AREA_NAME]:")
    print(gdf[["AREA_SHORT_CODE", "AREA_LONG_CODE", "AREA_NAME"]].head(5).to_string(index=False))
    return gdf


def load_profiles():
    path = _pick_one("*.xlsx", "profiles XLSX") or _pick_one("*.csv", "profiles CSV")
    if not path:
        return None
    print(f"\n  profiles: {os.path.basename(path)}")
    if path.lower().endswith(".xlsx"):
        xl = pd.ExcelFile(path)
        print(f"    sheets: {xl.sheet_names}")
        sheet = PROFILE_SHEET if PROFILE_SHEET in xl.sheet_names else xl.sheet_names[0]
        raw = xl.parse(sheet, header=None)
    else:
        raw = pd.read_csv(path, header=None, dtype=str)
    print(f"    raw shape: {raw.shape}  (wide: indicators as rows, areas as columns)")

    labels = raw.iloc[:, 0].astype(str).str.strip()
    name_idx = labels[labels == PROFILE_NAME_ROW].index
    num_idx = labels[labels == PROFILE_NUM_ROW].index
    if len(name_idx) == 0 or len(num_idx) == 0:
        print(f"    !! could not find '{PROFILE_NAME_ROW}' / '{PROFILE_NUM_ROW}' rows.")
        print(f"    first 6 row labels: {list(labels[:6])}")
        return None

    names = raw.iloc[name_idx[0], 1:].astype(str).str.strip()
    nums = raw.iloc[num_idx[0], 1:]
    prof = pd.DataFrame({
        "neighbourhood_number": pd.to_numeric(nums, errors="coerce").astype("Int64").values,
        "profile_name": names.values,
    })
    print(f"    parsed {len(prof)} neighbourhoods")
    print(f"    number range: min={prof['neighbourhood_number'].min()} "
          f"max={prof['neighbourhood_number'].max()}")
    print("    sample [neighbourhood_number, profile_name]:")
    print(prof.head(5).to_string(index=False))
    return prof


def check_join(gdf, prof) -> None:
    print("\n  ---- JOIN KEY CHECK  (AREA_SHORT_CODE  vs  Neighbourhood Number) ----")
    if gdf is None or prof is None or "AREA_SHORT_CODE" not in gdf.columns:
        print("    cannot check -- one side failed to load.")
        return
    b = pd.DataFrame({
        "code": gdf["AREA_SHORT_CODE"].astype(int).values,
        "boundary_name": gdf["AREA_NAME"].astype(str).str.strip().values,
    })
    p = prof.rename(columns={"neighbourhood_number": "code"}).dropna(subset=["code"])
    p["code"] = p["code"].astype(int)

    merged = b.merge(p, on="code", how="outer", indicator=True)
    both = (merged["_merge"] == "both").sum()
    only_b = (merged["_merge"] == "left_only").sum()
    only_p = (merged["_merge"] == "right_only").sum()
    print(f"    codes in both: {both} / {len(b)} boundaries, {len(p)} profiles")
    if only_b or only_p:
        print(f"    [WARN]  boundary-only codes: {only_b}   profile-only codes: {only_p}")
        print(merged[merged["_merge"] != "both"].to_string(index=False))
    else:
        print("    [OK] every boundary code has exactly one profile row and vice versa.")

    m = merged[merged["_merge"] == "both"].copy()
    m["name_match"] = m["boundary_name"].map(_norm) == m["profile_name"].map(_norm)
    bad = m[~m["name_match"]]
    print(f"    name agreement (punctuation-normalized): {m['name_match'].sum()} / {len(m)}")
    if len(bad):
        print("    [WARN]  codes where names differ even after normalization "
              "(investigate -- could be a real mis-map):")
        print(bad[["code", "boundary_name", "profile_name"]].to_string(index=False))

    raw_name_match = (m["boundary_name"] == m["profile_name"]).sum()
    print(f"    exact (raw) name agreement: {raw_name_match} / {len(m)} "
          f"-> JOIN ON THE NUMBER, not the name; carry AREA_NAME as the label.")


def inspect_boundaries_and_profiles() -> None:
    print("\n" + "=" * 78)
    print("2. Neighbourhood boundaries  vs  neighbourhood profiles")
    gdf = load_boundaries()
    prof = load_profiles()
    check_join(gdf, prof)


def main() -> int:
    if not os.path.isdir(RAW_DIR) or not glob.glob(os.path.join(RAW_DIR, "*")):
        print("data/raw/ is empty. Run: python scripts/00_fetch_data.py --download")
        return 1
    inspect_gtfs()
    inspect_boundaries_and_profiles()
    print("\n" + "=" * 78)
    print("Phase 0 validation done. Confirm the weekday service_id above, then Phase 1.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
