"""
Phase 0 (setup & validation) -- discover and fetch the three source datasets.

Inputs:  the City of Toronto CKAN API (`package_show`) for the three package ids
         in PACKAGES; nothing on disk.
Outputs: data/raw/opendata_ttc_schedules.zip                   (TTC GTFS feed)
         data/raw/neighbourhoods-4326.geojson                  (158 polygons)
         data/raw/nbhd_2021_census_profile_full_158model.xlsx  (2021 profiles)
         -- only with --download; a bare run just prints what it found.

What this does:
  1. Calls the CKAN `package_show` action for each of the three packages named
     in CLAUDE.md.
  2. Prints every resource it finds (name, format, id, last modified, URL) so a
     human can eyeball that the resources look right. NO filenames are assumed.
  3. Only when run with --download does it actually pull the files into
     data/raw/.

Usage:
    python scripts/00_fetch_data.py            # inspect only, download nothing
    python scripts/00_fetch_data.py --download # download after you've confirmed

Design notes / assumptions:
  - Each package exposes many resources (formats, CRS variants, and -- for
    neighbourhoods/profiles -- different vintages and id schemes). We do NOT
    hard-code resource filenames; instead SELECTION_RULES below picks one
    resource per package by keyword include/exclude/prefer rules, and the
    inspect run prints the full list plus *why* each pick was made so a human
    can confirm before --download.
  - We do NOT rename files on download; we keep whatever the portal calls them
    (falling back to the resource id if the URL has no usable basename), so
    data/raw/ stays a faithful copy of what was published.
  - Only ONE resource per package is downloaded (the confirmed "best" pick).
"""

from __future__ import annotations

import argparse
import os
import sys
from urllib.parse import urlparse

import requests

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

CKAN_BASE = "https://ckan0.cf.opendata.inter.prod-toronto.ca/api/3/action"
PACKAGE_SHOW = f"{CKAN_BASE}/package_show"

# Package ids exactly as recorded in CLAUDE.md section 4.
PACKAGES = {
    "gtfs": "ttc-routes-and-schedules",
    "boundaries": "neighbourhoods",
    "profiles": "neighbourhood-profiles",
}

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
RAW_DIR = os.path.abspath(RAW_DIR)

# Per-package resource-selection rules.
#
# These were tuned AFTER a first inspect run (python scripts/00_fetch_data.py
# with no flag) revealed the resource lists. Key facts that drive the rules:
#
#   * boundaries: the `neighbourhoods` package publishes BOTH the current 158
#     scheme and a "historical 140" scheme, in ~8 formats each, plus a
#     datastore-dump resource whose URL has no extension and defaults to CSV.
#     We want the current 158 polygons as a real .geojson file in EPSG:4326.
#   * profiles: the ONLY 2021 / 158-neighbourhood profile is an *.xlsx
#     (`...158model.xlsx`). Every CSV in that package is the older 2016/2011/
#     2006/2001 *140*-neighbourhood model -- wrong vintage AND wrong id scheme.
#     So for profiles we must take the XLSX, not a CSV.
#
# Matching is on the lowercased "name + url" of each resource.
SELECTION_RULES = {
    "gtfs": {
        "require_any": [".zip"],
        "exclude": [],
        "prefer": [".zip"],
    },
    "boundaries": {
        # current 158 scheme, real geojson file, WGS84
        "require_any": [".geojson", "geojson"],
        "exclude": ["historical", "140", "/datastore/dump/"],
        "prefer": ["4326.geojson", "4326", ".geojson"],
    },
    "profiles": {
        # 2021 census, 158-neighbourhood model (only exists as xlsx)
        "require_any": ["158", ".xlsx"],
        "exclude": ["140", "2016", "2011", "2006", "2001", ".xml", ".json"],
        "prefer": ["158model.xlsx", "158", ".xlsx"],
    },
}


def package_show(package_id: str) -> dict:
    resp = requests.get(PACKAGE_SHOW, params={"id": package_id}, timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    if not payload.get("success"):
        raise RuntimeError(f"CKAN reported failure for '{package_id}': {payload}")
    return payload["result"]


def describe_package(key: str, package_id: str) -> dict:
    print("=" * 78)
    print(f"[{key}] package id: {package_id}")
    result = package_show(package_id)
    print(f"  title:        {result.get('title')}")
    print(f"  licence:      {result.get('license_title')}")
    print(f"  last refresh: {result.get('metadata_modified')}")
    resources = result.get("resources", [])
    print(f"  {len(resources)} resource(s):")
    for i, r in enumerate(resources):
        print(f"    ---- resource #{i} ----")
        print(f"      name:          {r.get('name')}")
        print(f"      format:        {r.get('format')}")
        print(f"      id:            {r.get('id')}")
        print(f"      last modified: {r.get('last_modified') or r.get('metadata_modified')}")
        print(f"      datastore:     {r.get('datastore_active')}")
        print(f"      url:           {r.get('url')}")
    return result


def _haystack(r: dict) -> str:
    return f"{r.get('name') or ''} {r.get('url') or ''} {r.get('format') or ''}".lower()


def choose_resource(key: str, result: dict) -> tuple[dict | None, str]:
    """Pick one resource per package using SELECTION_RULES. Returns (resource, why)."""
    rules = SELECTION_RULES[key]
    resources = result.get("resources", [])

    candidates = []
    for r in resources:
        h = _haystack(r)
        if not r.get("url"):
            continue
        if rules["exclude"] and any(bad in h for bad in rules["exclude"]):
            continue
        if rules["require_any"] and not any(good in h for good in rules["require_any"]):
            continue
        candidates.append(r)

    if not candidates:
        return None, "no resource matched the selection rules"

    # Rank surviving candidates by first 'prefer' token they contain.
    def rank(r: dict) -> int:
        h = _haystack(r)
        for i, tok in enumerate(rules["prefer"]):
            if tok in h:
                return i
        return len(rules["prefer"])

    candidates.sort(key=rank)
    best = candidates[0]
    why = (f"matched require_any={rules['require_any']}, "
           f"passed exclude={rules['exclude']}, "
           f"ranked by prefer={rules['prefer']}")
    if len(candidates) > 1:
        why += f"  ({len(candidates)} candidates; others: " \
               + ", ".join(repr(c.get('name')) for c in candidates[1:4]) + ")"
    return best, why


def target_filename(resource: dict) -> str:
    """Keep the portal's own filename; fall back to <resource-id>.<ext>."""
    url = resource.get("url") or ""
    base = os.path.basename(urlparse(url).path)
    if base and "." in base:
        return base
    ext = (resource.get("format") or "bin").strip().lower()
    return f"{resource.get('id', 'resource')}.{ext}"


def download(resource: dict, dest_dir: str) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    fname = target_filename(resource)
    dest = os.path.join(dest_dir, fname)
    url = resource["url"]
    print(f"  downloading {url}")
    print(f"          ->  {dest}")
    with requests.get(url, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0))
        written = 0
        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                written += len(chunk)
                if total:
                    pct = written / total * 100
                    print(f"\r          {written/1e6:8.1f} MB / {total/1e6:.1f} MB ({pct:5.1f}%)",
                          end="", flush=True)
        if total:
            print()
    print(f"  done: {written/1e6:.1f} MB")
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--download", action="store_true",
                        help="actually download the chosen resource per package into data/raw/")
    args = parser.parse_args()

    results = {}
    for key, pkg_id in PACKAGES.items():
        try:
            results[key] = describe_package(key, pkg_id)
        except Exception as exc:  # noqa: BLE001 - surface any fetch problem clearly
            print(f"  ERROR describing '{pkg_id}': {exc}")
            results[key] = None

    print("=" * 78)
    print("AUTO-PICKED resource per package (what --download would fetch):")
    picks = {}
    for key, result in results.items():
        if not result:
            print(f"  [{key}] no package metadata -- skipped")
            continue
        pick, why = choose_resource(key, result)
        picks[key] = pick
        if pick:
            print(f"  [{key}] -> {pick.get('name')}  ({pick.get('format')})")
            print(f"           {pick.get('url')}")
            print(f"           why: {why}")
        else:
            print(f"  [{key}] no downloadable resource found  ({why})")

    if not args.download:
        print("\nInspect the resources above. If they look right, re-run with --download.")
        return 0

    print("\n" + "=" * 78)
    print(f"DOWNLOADING into {RAW_DIR}")
    for key, pick in picks.items():
        if not pick:
            print(f"  [{key}] nothing to download")
            continue
        print(f"\n[{key}]")
        try:
            download(pick, RAW_DIR)
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR downloading [{key}]: {exc}")
    print("\nDone. Raw files are in data/raw/ (leave them untouched -- see CLAUDE.md).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
