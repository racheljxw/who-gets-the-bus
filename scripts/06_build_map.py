"""
Phase 4 -- build the interactive folium map (output/index.html).

Input:
  data/processed/equity.geojson   (Phase 3, 158 polygons EPSG:4326, scores locked)
  data/processed/access.parquet   (Phase 1b/patches -- only for total_effective_frequency,
                                   the capacity-weighted + walkshed frequency column that
                                   Phase 3 did NOT carry into equity.geojson)

Output:
  output/index.html               self-contained Leaflet map (folium embeds its own JS/CSS)

What the map has (CLAUDE.md section 1):
  * one base map centered on Toronto
  * THREE choropleth layers, radio-select (overlay=False -> LayerControl renders them as
    a single-choice group so only one is ever painted):
       - "Transit Equity Gap"  equity_gap   diverging RdYlBu_r, symmetric bins centered at 0
                                            -- shown by default
       - "Transit Access"      access_score sequential YlOrRd, 0 -> max
       - "Need"                need_score   sequential YlOrRd, 0 -> max
  * ONE always-on transparent GeoJson layer on top carrying the hover tooltip, so the same
    per-neighbourhood readout works no matter which choropleth is active
  * a LayerControl, a title card, and a licence-attribution footer

Run:
    venv/Scripts/python scripts/06_build_map.py
"""

from __future__ import annotations

import os
import sys

import re
import urllib.request

import json

import branca.colormap as cm
import folium
import geopandas as gpd
import numpy as np
import pandas as pd
from branca.element import MacroElement
from folium.features import GeoJsonTooltip
from jinja2 import Template

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

HERE = os.path.dirname(__file__)
EQUITY_GEOJSON = os.path.abspath(os.path.join(HERE, "..", "data", "processed", "equity.geojson"))
ACCESS_PARQUET = os.path.abspath(os.path.join(HERE, "..", "data", "processed", "access.parquet"))
OUT_HTML = os.path.abspath(os.path.join(HERE, "..", "output", "index.html"))

TORONTO_CENTER = [43.72, -79.37]   # roughly the centroid of the 158-neighbourhood extent
DEFAULT_ZOOM = 11                  # shows all of Toronto with a little margin

# Data snapshot / GTFS validity -- from CLAUDE.md section 9 (Phase 0 findings).
GTFS_VALID = "TTC GTFS feed valid 2026-09-06 to 2026-10-31"
DATA_SNAPSHOT = "Open data retrieved 2026-09-01"

# Basemap: Esri "World Light Gray" (base + a labels layer that rides ON TOP of the
# choropleth so place names stay readable). Keyless, muted -- CartoDB Positron, the
# usual choice, now serves an "API KEY REQUIRED" watermark for keyless use.
ESRI_BASE_URL = ("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/"
                 "World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}")
ESRI_REF_URL = ("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/"
                "World_Light_Gray_Reference/MapServer/tile/{z}/{y}/{x}")
ESRI_ATTR = ("Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ, "
             "&copy; OpenStreetMap contributors")

LICENCE_HTML = (
    "Data: "
    "<a href='https://open.toronto.ca/dataset/ttc-routes-and-schedules/' target='_blank' "
    "rel='noopener'>TTC Routes &amp; Schedules (GTFS)</a>, "
    "<a href='https://open.toronto.ca/dataset/neighbourhoods/' target='_blank' "
    "rel='noopener'>Neighbourhood Boundaries (158)</a> and "
    "<a href='https://open.toronto.ca/dataset/neighbourhood-profiles/' target='_blank' "
    "rel='noopener'>Neighbourhood Profiles (2021 Census)</a> "
    "&mdash; City of Toronto Open Data (open.toronto.ca), "
    "<a href='https://open.toronto.ca/open-data-license/' target='_blank' "
    "rel='noopener'>Open Government Licence &ndash; Toronto</a>. "
    f"{GTFS_VALID}. {DATA_SNAPSHOT}. "
    f"Basemap: {ESRI_ATTR}."
)


def inline_web_assets(html: str) -> str:
    """Fetch every CDN <script src>/<link href> once and inline it, so the saved
    HTML renders with no network except the basemap tiles themselves (a web map
    always streams those). Folium 0.20 links these from CDNs by default; the
    project ships as static files, so we vendor them into the one file.
    A fetch failure leaves that tag as-is and prints a warning."""
    cache = os.path.join(
        os.environ.get("TEMP", "/tmp"), "wgtb_webassets"
    )
    os.makedirs(cache, exist_ok=True)

    def _get(url: str) -> str | None:
        key = os.path.join(cache, re.sub(r"[^A-Za-z0-9._-]", "_", url))
        if os.path.isfile(key):
            return open(key, encoding="utf-8").read()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "wgtb-build"})
            with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310
                body = r.read().decode("utf-8", "replace")
            open(key, "w", encoding="utf-8").write(body)
            return body
        except Exception as e:  # noqa: BLE001
            print(f"  WARN could not vendor {url}: {e}")
            return None

    for m in re.finditer(r'<script src="(https?://[^"]+)"></script>', html):
        body = _get(m.group(1))
        if body is not None:
            html = html.replace(m.group(0), f"<script>\n{body}\n</script>")
    for m in re.finditer(r'<link rel="stylesheet" href="(https?://[^"]+)"/?>', html):
        body = _get(m.group(1))
        if body is not None:
            html = html.replace(m.group(0), f"<style>\n{body}\n</style>")
    return html


def _fmt(v, spec):
    """Format a scalar, but pass NaN through as an em dash so the tooltip stays readable."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return format(v, spec)


def main() -> int:
    for p in (EQUITY_GEOJSON, ACCESS_PARQUET):
        if not os.path.isfile(p):
            print(f"missing {p}")
            return 1

    gdf = gpd.read_file(EQUITY_GEOJSON)
    assert len(gdf) == 158, f"expected 158 features, got {len(gdf)}"
    assert gdf.crs is not None and gdf.crs.to_epsg() == 4326, f"expected EPSG:4326, got {gdf.crs}"
    gdf["AREA_SHORT_CODE"] = gdf["AREA_SHORT_CODE"].astype(int)

    # The map embeds the geometry once per layer (3 choropleths + 1 tooltip layer).
    # Simplify by ~15 m in a projected CRS -- invisible at city zoom -- so the
    # self-contained HTML stays a couple of MB instead of ~8. Scores are untouched.
    gdf["geometry"] = (
        gdf.geometry.to_crs(2952).simplify(15, preserve_topology=True).to_crs(4326)
    )

    # ---- bring in the one frequency column Phase 3 left in access.parquet ----
    # equity.geojson carries only the plain audit count (neighbourhood_frequency).
    # For the tooltip we want the locked capacity-weighted + walkshed number that
    # actually drives the access score: total_effective_frequency.
    freq = pd.read_parquet(ACCESS_PARQUET)[
        ["AREA_SHORT_CODE", "total_effective_frequency"]
    ].copy()
    freq["AREA_SHORT_CODE"] = freq["AREA_SHORT_CODE"].astype(int)
    gdf = gdf.merge(freq, on="AREA_SHORT_CODE", how="left")
    assert gdf["total_effective_frequency"].notna().all(), "frequency join left gaps"
    assert len(gdf) == 158

    # ---- pre-format tooltip fields (GeoJsonTooltip cannot format numbers itself) ----
    gdf["tt_gap"] = gdf["equity_gap"].map(lambda v: _fmt(v, "+.3f"))
    gdf["tt_access"] = gdf["access_score"].map(lambda v: _fmt(v, ".3f"))
    gdf["tt_need"] = gdf["need_score"].map(lambda v: _fmt(v, ".3f"))
    gdf["tt_stops"] = gdf["stop_count"].astype(int).astype(str)
    gdf["tt_freq"] = gdf["total_effective_frequency"].map(lambda v: _fmt(v, ",.1f"))
    gdf["tt_lowinc"] = gdf["low_income_pct"].map(lambda v: _fmt(v, ".1f") + "%")
    gdf["tt_noncar"] = gdf["non_car_commute_pct"].map(lambda v: _fmt(v, ".1f") + "%")
    gdf["tt_dens"] = gdf["population_density"].map(lambda v: _fmt(v, ",.0f"))

    gap_abs = float(np.abs(gdf["equity_gap"]).max())
    acc_max = float(gdf["access_score"].max())
    need_max = float(gdf["need_score"].max())

    # ---- base map ----
    # tiles=None + explicit control=False TileLayers: otherwise folium files the
    # basemap as a base layer too, and the radio group (see below) would let the
    # user switch the map background off.
    m = folium.Map(
        location=TORONTO_CENTER,
        zoom_start=DEFAULT_ZOOM,
        tiles=None,
        control_scale=True,
        # default Leaflet attribution box sits under our footer; all credits
        # (data licences + basemap) are in the footer instead.
        attributionControl=False,
    )
    folium.TileLayer(
        tiles=ESRI_BASE_URL, attr=ESRI_ATTR, name="basemap",
        control=False, max_zoom=16,
    ).add_to(m)

    # ---- colour scales (branca colormaps, used ONLY to colour the polygons) ----
    # Diverging, symmetric about 0: negative = over-served (blue), positive = underserved (red).
    YLORRD = ["#ffffcc", "#ffeda0", "#fed976", "#feb24c", "#fd8d3c",
              "#fc4e2a", "#e31a1c", "#bd0026", "#800026"]
    GAP_COLORS = ["#4575b4", "#91bfdb", "#e0f3f8", "#fee090", "#fc8d59", "#d73027"]

    gap_bin = round(gap_abs + 0.02, 2)  # pad so the extreme value sits inside the outer bin
    gap_bins = [round(x, 4) for x in np.linspace(-gap_bin, gap_bin, 7)]
    gap_cmap = cm.StepColormap(GAP_COLORS, vmin=-gap_bin, vmax=gap_bin, index=gap_bins)
    acc_cmap = cm.LinearColormap(YLORRD, vmin=0.0, vmax=acc_max)
    need_cmap = cm.LinearColormap(YLORRD, vmin=0.0, vmax=need_max)

    def _style(cmap, col):
        def style_function(feat):
            v = feat["properties"][col]
            return {
                "fillColor": cmap(v) if v is not None else "#cccccc",
                "color": "#555555",
                "weight": 0.5,
                "fillOpacity": 0.72,
            }
        return style_function

    # ---- three choropleth layers, radio-select (overlay=False) ----
    # We hand-roll the fill + legend (folium.GeoJson + branca colormap) rather than
    # folium.Choropleth: Choropleth's d3 legend has a selector bug when several sit on
    # one map (every SVG lands in the first legend box), and it will not centre a
    # diverging ramp on 0. Same visual result, full control of the diverging domain.
    layers = [
        ("Transit Equity Gap", "equity_gap", gap_cmap, True),
        ("Transit Access", "access_score", acc_cmap, False),
        ("Need", "need_score", need_cmap, False),
    ]
    layer_names = {}
    for name, col, cmap, show in layers:
        gj = folium.GeoJson(
            gdf.__geo_interface__,
            name=name,
            overlay=False,      # -> LayerControl treats these as a single-choice group
            show=show,
            style_function=_style(cmap, col),
            highlight_function=lambda _f: {"weight": 2, "color": "#111111", "fillOpacity": 0.9},
            smooth_factor=0.5,
        )
        gj.add_to(m)
        layer_names[name] = gj.get_name()

    # Esri place-name labels. Leaflet keeps all tile layers in tilePane (below the
    # SVG overlayPane), so these sit *under* the choropleth, but the 0.72 fill
    # opacity lets major labels (Toronto, North York, Scarborough, …) read through.
    folium.TileLayer(
        tiles=ESRI_REF_URL, attr=ESRI_ATTR, name="labels",
        control=False, max_zoom=16, overlay=True,
    ).add_to(m)

    # ---- one always-on transparent tooltip layer on top ----
    tooltip = GeoJsonTooltip(
        fields=["AREA_NAME", "tt_gap", "tt_access", "tt_need", "tt_stops",
                "tt_freq", "tt_lowinc", "tt_noncar", "tt_dens"],
        aliases=["Neighbourhood", "Equity gap (need − access)", "Access score",
                 "Need score", "TTC stops in neighbourhood",
                 "Capacity-weighted trips/hr (incl. walkshed)", "Low-income (LIM-AT)",
                 "Non-car commute share", "Population density (people/km²)"],
        localize=False,
        sticky=True,
        labels=True,
        style=(
            "background-color:#ffffff; color:#222; font-family:system-ui,sans-serif; "
            "font-size:12px; padding:8px 10px; border:1px solid #bbb; border-radius:4px; "
            "box-shadow:0 1px 4px rgba(0,0,0,0.25);"
        ),
    )
    folium.GeoJson(
        gdf.__geo_interface__,
        name="Neighbourhood details (hover)",
        control=False,          # always on, not a toggle
        style_function=lambda _f: {"fillColor": "#000000", "color": "#000000",
                                   "weight": 0, "fillOpacity": 0},
        highlight_function=lambda _f: {"weight": 2.5, "color": "#111111", "fillOpacity": 0.06},
        tooltip=tooltip,
        smooth_factor=0.5,
    ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # ---- one legend per layer; JS shows only the active layer's legend ----
    def _bar(colors):
        segs = "".join(
            f"<span style='flex:1 1 0;height:14px;background:{c};'></span>" for c in colors
        )
        return (f"<span style='display:flex;width:190px;border:1px solid #999;'>"
                f"{segs}</span>")

    def _legend(lid, title, colors, left, mid, right, note, shown):
        mid_html = (f"<span style='flex:1;text-align:center;'>{mid}</span>"
                    if mid is not None else "")
        return (
            f"<div id='{lid}' class='wgtb-legend' style=\"display:{'block' if shown else 'none'};"
            "position:fixed; right:12px; bottom:34px; z-index:9999; "
            "background:rgba(255,255,255,0.94); border:1px solid #bbb; border-radius:5px; "
            "padding:7px 10px; font-family:system-ui,sans-serif; font-size:11px; color:#222;\">"
            f"<div style='font-weight:700; margin-bottom:4px;'>{title}</div>"
            f"{_bar(colors)}"
            "<div style='display:flex; width:190px; font-size:10px; margin-top:2px;'>"
            f"<span style='flex:1;'>{left}</span>{mid_html}"
            f"<span style='flex:1; text-align:right;'>{right}</span></div>"
            f"<div style='font-size:10px; color:#555; margin-top:3px; max-width:190px;'>{note}</div>"
            "</div>"
        )

    slug = {"Transit Equity Gap": "gap", "Transit Access": "access", "Need": "need"}
    legends_html = (
        _legend("wgtb-legend-gap", "Transit equity gap (need − access)",
                GAP_COLORS, f"{-gap_bin:+.2f}", "0", f"{gap_bin:+.2f}",
                "Blue = better served than need suggests · Red = underserved", True)
        + _legend("wgtb-legend-access", "Transit access score",
                  YLORRD, "0.00", None, f"{acc_max:.2f}",
                  "Stop density + capacity-weighted peak frequency (incl. 500 m rapid-transit walkshed). Higher = better served.", False)
        + _legend("wgtb-legend-need", "Transit need score",
                  YLORRD, "0.00", None, f"{need_max:.2f}",
                  "Mean of low-income share, non-car-commute share, population density. Higher = more transit-dependent.", False)
    )
    m.get_root().html.add_child(folium.Element(legends_html))

    # Show only the active layer's legend. A MacroElement (not a bare script child)
    # so folium renders this INSIDE the map's script block, after the layers/control
    # exist -- a plain figure-level script would run before `map_...` is defined.
    class LegendToggle(MacroElement):
        _template = Template(
            "{% macro script(this, kwargs) %}\n"
            "var __slug = " + json.dumps(slug) + ";\n"
            "function __showLegend(nm){\n"
            "  document.querySelectorAll('.wgtb-legend').forEach(function(d){d.style.display='none';});\n"
            "  var el = document.getElementById('wgtb-legend-'+(__slug[nm]||nm));\n"
            "  if(el) el.style.display='block';\n"
            "}\n"
            "{{ this._parent.get_name() }}.on('baselayerchange', function(e){ __showLegend(e.name); });\n"
            "{% endmacro %}"
        )

    m.add_child(LegendToggle())

    # ---- title card + licence footer (plain HTML on the page) ----
    title_html = (
        "<div style=\"position:fixed; top:12px; left:50px; z-index:9999; "
        "background:rgba(255,255,255,0.92); padding:8px 14px; border:1px solid #bbb; "
        "border-radius:5px; font-family:system-ui,sans-serif; max-width:420px;\">"
        "<div style=\"font-size:15px; font-weight:700;\">Who Gets the Bus?</div>"
        "<div style=\"font-size:12px; color:#444;\">Toronto transit equity gap by "
        "neighbourhood &mdash; need minus access, weekday morning peak. "
        "Toggle layers at right; hover any neighbourhood for detail.</div></div>"
    )
    footer_html = (
        "<div style=\"position:fixed; bottom:0; left:0; right:0; z-index:9999; "
        "background:rgba(255,255,255,0.92); border-top:1px solid #bbb; padding:4px 10px; "
        "font-family:system-ui,sans-serif; font-size:10.5px; color:#333; line-height:1.35;\">"
        f"{LICENCE_HTML}</div>"
    )
    m.get_root().html.add_child(folium.Element(title_html))
    m.get_root().html.add_child(folium.Element(footer_html))

    os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
    html = m.get_root().render()
    html = inline_web_assets(html)
    with open(OUT_HTML, "w", encoding="utf-8") as fh:
        fh.write(html)

    remaining = re.findall(r'(?:src|href)="(https?://(?![^"]*basemaps|[^"]*tile)[^"]+)"', html)
    if remaining:
        print(f"  NOTE {len(remaining)} external ref(s) still linked: {remaining}")

    size_kb = os.path.getsize(OUT_HTML) / 1024
    print(f"wrote {OUT_HTML}  ({size_kb:,.0f} KB)")
    print(f"  layers        : {[n for n, *_ in layers]} (radio, 'Transit Equity Gap' default)")
    print(f"  gap bins      : {gap_bins}  (symmetric, centered 0; |max gap| = {gap_abs:.3f})")
    print(f"  access domain : 0 .. {acc_max:.3f}")
    print(f"  need domain   : 0 .. {need_max:.3f}")
    print(f"  tooltip fields: AREA_NAME, equity_gap, access_score, need_score, stop_count,")
    print(f"                  total_effective_frequency, low_income_pct, non_car_commute_pct,")
    print(f"                  population_density")
    return 0


if __name__ == "__main__":
    sys.exit(main())
