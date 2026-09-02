"""
Phase 4 -- build the interactive folium map (output/index.html).
Phase 4 UI patch (2026-09-02) -- folium's default chrome is replaced with a custom
dark UI: a page shell (title block + CSS grid), a hand-built layers control, a
single Legend panel, and a custom zoom pill. folium still renders the Leaflet map
and the choropleth / tooltip layers; everything visual around it is post-processed
onto the rendered HTML (see build_shell / SHELL_CSS / SHELL_JS below). No scoring
or data logic changed.

Input:
  data/processed/equity.geojson   (Phase 3, 158 polygons EPSG:4326, scores locked)
  data/processed/access.parquet   (Phase 1b/patches -- only for total_effective_frequency,
                                   the capacity-weighted + walkshed frequency column that
                                   Phase 3 did NOT carry into equity.geojson)

Output:
  output/index.html               self-contained Leaflet map (folium embeds its own JS/CSS)

What the map has (CLAUDE.md section 1):
  * one base map centered on Toronto (Esri dark-gray canvas -- keyless, matches the dark UI)
  * THREE choropleth layers, one painted at a time -- switched by the custom Layers control,
    NOT folium's LayerControl (removed):
       - "Equity Gap"  equity_gap   diverging sky-blue->hot-pink, symmetric bins centred on 0
                                     -- shown by default
       - "Access"      access_score sequential YlOrRd, 0 -> max
       - "Need"        need_score   sequential YlOrRd, 0 -> max
  * ONE always-on transparent GeoJson layer on top carrying the hover tooltip (unchanged),
    kept in front on every layer switch via bringToFront()
  * a title block, a Legend panel that swaps with the active layer, a custom zoom control,
    and a licence-attribution footer

Run:
    venv/Scripts/python scripts/06_build_map.py
"""

from __future__ import annotations

import base64
import json
import os
import re
import sys
import urllib.request

import branca.colormap as cm
import folium
import geopandas as gpd
import numpy as np
import pandas as pd
from folium.features import GeoJsonTooltip

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

# Basemap: Esri "Dark Gray Canvas" (base + a labels layer that rides ON TOP of the
# choropleth so place names stay readable). Keyless like the light-gray canvas the
# earlier build used -- swapped to the dark variant so the map reads with the dark
# UI shell instead of glowing white inside it. CartoDB dark_matter, the other usual
# pick, now serves an "API KEY REQUIRED" watermark for keyless use.
ESRI_BASE_URL = ("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/"
                 "World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}")
ESRI_REF_URL = ("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/"
                "World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}")
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

# ---- palette (confirmed design 2026-09-01) -------------------------------------
C_BG = "#0A0428"          # page background
C_PANEL = "#221C3F"       # panel background
C_PANEL_BORDER = "#C6C0E6"
C_MUTED = "#9286D0"       # inactive dot outline / subtitle / secondary text
C_MUTED_2 = "#C6C0E6"     # primary panel text
C_ACTIVE_FILL = "#24166D"
C_ACTIVE_BORDER = "#D9D2FF"
C_TEXT_ACTIVE = "#f4f4f8"

# 6-step diverging legend/choropleth ramp (blue = over-served, pink = underserved).
# Neon deep-indigo -> hot-pink (2026-09-02 tweak). The blue half is a smooth
# interpolation out of the #3E15C6 endpoint (was a jump straight to cyan); the pink
# half is unchanged. Same semantics, brighter on the dark basemap. Applied to BOTH
# the gap choropleth and its legend. Access/Need keep the sequential YlOrRd ramp.
GAP_COLORS = ["#3E15C6", "#7B5FDB", "#B7A9F0", "#ffafcc", "#ff5da2", "#ff006e"]
GAP_KEY_BLUE = "#7B5FDB"   # legend key swatch / callout for the over-served end
GAP_KEY_PINK = "#ff5da2"   # legend key swatch / callout for the underserved end
YLORRD = ["#ffffcc", "#ffeda0", "#fed976", "#feb24c", "#fd8d3c",
          "#fc4e2a", "#e31a1c", "#bd0026", "#800026"]

GOOGLE_FONT_URL = ("https://fonts.googleapis.com/css2?"
                   "family=Noto+Sans:wght@400;500;700&display=swap")

_CACHE = os.path.join(os.environ.get("TEMP", "/tmp"), "wgtb_webassets")
_CHROME_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


def _download(url: str, ua: str = "wgtb-build") -> bytes | None:
    """Fetch a URL once, disk-cached, returning raw bytes (or None on failure)."""
    os.makedirs(_CACHE, exist_ok=True)
    key = os.path.join(_CACHE, re.sub(r"[^A-Za-z0-9._-]", "_", ua + "|" + url)[:180])
    if os.path.isfile(key):
        return open(key, "rb").read()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": ua})
        with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310
            body = r.read()
        open(key, "wb").write(body)
        return body
    except Exception as e:  # noqa: BLE001
        print(f"  WARN could not fetch {url}: {e}")
        return None


def vendor_google_font(html: str) -> str:
    """Replace the fonts.googleapis.com <link> with an inline <style> whose
    @font-face src is a base64 data: URI per weight, so the published page pulls
    Noto Sans with zero runtime third-party fetches (CLAUDE.md section 2). A
    Chrome UA is needed for googleapis to answer with woff2. On any failure the
    <link> is left in place (runtime fallback) with a warning."""
    m = re.search(r'<link[^>]+href="(https://fonts\.googleapis\.com/[^"]+)"[^>]*>', html)
    if not m:
        return html
    css_bytes = _download(m.group(1), ua=_CHROME_UA)
    if css_bytes is None:
        return html
    css = css_bytes.decode("utf-8", "replace")
    # Noto Sans ships ~8 unicode subsets per weight; the page is English + a few
    # punctuation/symbol glyphs, so keep only the latin blocks -- inlining all of
    # them adds ~2 MB of base64 for glyphs that never render.
    keep = []
    for block in re.split(r"(?=/\* [\w-]+ \*/)", css):
        label = re.match(r"/\* ([\w-]+) \*/", block.strip())
        if label and label.group(1) not in ("latin", "latin-ext"):
            continue
        keep.append(block)
    css = "".join(keep)
    for fm in re.finditer(r'url\((https://fonts\.gstatic\.com/[^)]+)\)', css):
        raw = _download(fm.group(1), ua=_CHROME_UA)
        if raw is None:
            continue
        b64 = base64.b64encode(raw).decode("ascii")
        css = css.replace(fm.group(1), f"data:font/woff2;base64,{b64}")
    return html.replace(m.group(0), f"<style>\n{css}\n</style>")


def inline_web_assets(html: str) -> str:
    """Fetch every CDN <script src>/<link href> once and inline it, so the saved
    HTML renders with no network except the basemap tiles themselves (a web map
    always streams those). Folium 0.20 links these from CDNs by default; the
    project ships as static files, so we vendor them into the one file.
    A fetch failure leaves that tag as-is and prints a warning."""
    for m in re.finditer(r'<script src="(https?://[^"]+)"></script>', html):
        body = _download(m.group(1))
        if body is not None:
            html = html.replace(m.group(0), f"<script>\n{body.decode('utf-8', 'replace')}\n</script>")
    for m in re.finditer(r'<link rel="stylesheet" href="(https?://[^"]+)"/?>', html):
        body = _download(m.group(1))
        if body is not None:
            html = html.replace(m.group(0), f"<style>\n{body.decode('utf-8', 'replace')}\n</style>")
    return html


def _fmt(v, spec):
    """Format a scalar, but pass NaN through as an em dash so the tooltip stays readable."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return format(v, spec)


# -----------------------------------------------------------------------------
#  Custom dark-UI shell -- CSS, page scaffold, and the layer/zoom wiring JS.
#  These are string-injected onto folium's rendered HTML (folium 0.20 has no
#  templating hook for a surrounding page layout, and its LayerControl markup is
#  not reusable for this design -- see build_map notes at the bottom of the file).
# -----------------------------------------------------------------------------

SHELL_CSS = f"""
<style>
  :root {{
    --bg:{C_BG}; --panel:{C_PANEL}; --panel-border:{C_PANEL_BORDER};
    --muted:{C_MUTED}; --muted-2:{C_MUTED_2};
    --active-fill:{C_ACTIVE_FILL}; --active-border:{C_ACTIVE_BORDER};
    --text-active:{C_TEXT_ACTIVE};
  }}
  * {{ box-sizing:border-box; }}
  html, body {{
    width:100%; min-height:100%; margin:0; padding:0;
    background:var(--bg); color:var(--muted-2);
    font-family:"Noto Sans", sans-serif;
  }}
  /* folium ships html,body{{height:100%}} and #map{{position:absolute;inset:0}};
     override so the map lives inside our panel instead of filling the viewport. */
  .folium-map {{
    position:absolute !important; inset:0 !important;
    width:100% !important; height:100% !important;
  }}
  /* Noto Sans everywhere -- incl. Leaflet's own chrome and the hover tooltip,
     whose inline style would otherwise fall back to system-ui. */
  .leaflet-container {{ background:var(--panel); }}
  .leaflet-container, .leaflet-container *,
  .leaflet-tooltip, .leaflet-popup, .leaflet-control {{
    font-family:"Noto Sans", sans-serif !important;
  }}
  /* Leaflet leaves a browser focus ring on an SVG path after a click -- it
     renders as a black bounding-box rectangle around the neighbourhood. */
  .leaflet-interactive:focus,
  .leaflet-interactive:focus-visible,
  .leaflet-container svg path:focus {{ outline:none; }}

  .wgtb-app {{
    display:flex; flex-direction:column; gap:14px;
    padding:22px 64px; min-height:100vh;   /* wide side margins on laptop; trimmed on mobile */
  }}
  .wgtb-title-main {{ font-size:22px; font-weight:600; color:var(--muted-2); }}

  .wgtb-grid {{
    display:grid; grid-template-columns:1fr 200px; gap:14px;
    flex:1 1 auto; min-height:0;
  }}
  .wgtb-map-panel {{
    position:relative; overflow:hidden;
    background:var(--panel); border:1px solid var(--panel-border); border-radius:6px;
    min-height:460px;
  }}
  .wgtb-side {{ display:flex; flex-direction:column; gap:14px; }}
  .wgtb-panel {{
    background:var(--panel); border:1px solid var(--panel-border);
    border-radius:6px; padding:12px;
  }}
  .wgtb-panel-h {{
    font-size:13px; font-weight:500; color:var(--muted-2); margin-bottom:9px;
  }}

  /* Layers -- desktop: stacked rectangles */
  .wgtb-layer-list {{ display:flex; flex-direction:column; gap:7px; }}
  .wgtb-layer {{
    display:flex; align-items:center; gap:8px; width:100%;
    padding:7px 9px; border-radius:4px;
    font-family:inherit; font-size:13px; text-align:left; cursor:pointer;
    background:transparent; border:1px solid var(--muted); color:var(--muted-2);
  }}
  .wgtb-layer .wgtb-dot {{
    width:9px; height:9px; border-radius:50%; flex:0 0 auto;
    background:transparent; border:1px solid var(--muted);
  }}
  .wgtb-layer.is-active {{
    background:var(--active-fill); border-color:var(--active-border);
    color:var(--text-active);
  }}
  .wgtb-layer.is-active .wgtb-dot {{
    background:var(--active-border); border-color:var(--active-border);
  }}

  /* Legend */
  .wgtb-legend-bar {{
    display:flex; width:100%; height:12px; border-radius:2px; overflow:hidden;
  }}
  .wgtb-legend-bar span {{ flex:1 1 0; }}
  .wgtb-legend-ends {{
    display:flex; justify-content:space-between;
    font-size:10px; color:var(--muted); margin-top:4px;
  }}
  .wgtb-legend-key {{
    display:flex; align-items:center; gap:6px;
    font-size:10.5px; color:var(--muted-2); margin-top:6px; line-height:1.35;
  }}
  .wgtb-legend-key i {{
    width:10px; height:10px; border-radius:2px; flex:0 0 auto; display:inline-block;
  }}
  .wgtb-legend-note {{
    font-size:10px; color:var(--muted); margin-top:6px; line-height:1.35;
  }}

  /* Custom zoom control -- joined pill, bottom-right inside the map panel */
  .wgtb-zoom {{
    position:absolute; right:12px; bottom:12px; z-index:1000;
    display:flex; border-radius:4px; overflow:hidden;
    background:var(--panel); border:1px solid var(--panel-border);
  }}
  .wgtb-zoom button {{
    width:28px; height:28px; padding:0; line-height:1;
    display:flex; align-items:center; justify-content:center;
    font-family:inherit; font-size:16px; cursor:pointer;
    background:transparent; border:0; color:var(--muted-2);
  }}
  .wgtb-zoom button:first-child {{ border-right:1px solid var(--panel-border); }}
  .wgtb-zoom button:hover {{ background:var(--active-fill); color:var(--text-active); }}

  .wgtb-footer {{ font-size:10px; color:var(--muted); line-height:1.4; }}
  .wgtb-footer a {{ color:var(--muted-2); }}

  @media (max-width:768px) {{
    /* flatten the nesting so title / layers / map / legend / footer are all
       direct flex items of .wgtb-app and can be re-ordered vertically */
    .wgtb-grid, .wgtb-side {{ display:contents; }}
    .wgtb-app        {{ padding:16px; }}
    .wgtb-title      {{ order:1; }}
    .wgtb-layers     {{ order:2; }}
    .wgtb-map-panel  {{ order:3; min-height:60vh; }}
    .wgtb-legend-panel {{ order:4; }}
    .wgtb-footer     {{ order:5; }}

    .wgtb-title-main {{ font-size:18px; }}

    /* Layers -- mobile: separate pill buttons in a row, thinner than desktop rows.
       The radio dot is desktop-only; the pills read as a segmented control. */
    .wgtb-layer-list {{ flex-direction:row; gap:8px; }}
    .wgtb-layer {{
      flex:1 1 0; justify-content:center; gap:0;
      padding:5px 8px; border-radius:999px;
      background:var(--panel); border-color:var(--muted);
    }}
    .wgtb-layer .wgtb-dot {{ display:none; }}
    .wgtb-layer.is-active {{ background:var(--active-fill); border-color:var(--active-border); }}
  }}
</style>
"""


def build_shell(map_name: str, layer_js: dict, tooltip_js: str,
                legends: dict, footer_html: str, data_bounds: tuple) -> tuple[str, str]:
    """Return (scaffold_html, wiring_js). scaffold_html wraps the folium-map div;
    wiring_js is appended after folium's own <script>."""
    rows = "".join(
        f'<button type="button" class="wgtb-layer{" is-active" if key == "gap" else ""}" '
        f'data-layer="{key}"><span class="wgtb-dot"></span><span>{label}</span></button>'
        for key, label in (("gap", "Equity gap"), ("access", "Access"), ("need", "Need"))
    )
    scaffold = (
        '<div class="wgtb-app">'
        '  <header class="wgtb-title">'
        '    <div class="wgtb-title-main">Toronto Transit Equity Map</div>'
        '  </header>'
        '  <div class="wgtb-grid">'
        '    <div class="wgtb-map-panel">'
        f'      <div class="folium-map" id="{map_name}"></div>'
        '      <div class="wgtb-zoom">'
        '        <button type="button" class="wgtb-zoom-out" aria-label="Zoom out">&#8722;</button>'
        '        <button type="button" class="wgtb-zoom-in" aria-label="Zoom in">+</button>'
        '      </div>'
        '    </div>'
        '    <aside class="wgtb-side">'
        '      <div class="wgtb-panel wgtb-layers">'
        '        <div class="wgtb-panel-h">Layers</div>'
        f'        <div class="wgtb-layer-list">{rows}</div>'
        '      </div>'
        '      <div class="wgtb-panel wgtb-legend-panel">'
        '        <div class="wgtb-panel-h">Legend</div>'
        '        <div class="wgtb-legend-body"></div>'
        '      </div>'
        '    </aside>'
        '  </div>'
        f'  <footer class="wgtb-footer">{footer_html}</footer>'
        '</div>'
    )

    wiring = (
        "<script>\n"
        "(function () {\n"
        f"  var MAP = {map_name};\n"
        f"  var LAYERS = {{ gap: {layer_js['gap']}, access: {layer_js['access']}, "
        f"need: {layer_js['need']} }};\n"
        f"  var TT = {tooltip_js};\n"
        f"  var LEGENDS = {json.dumps(legends)};\n"
        "  var body = document.querySelector('.wgtb-legend-body');\n"
        "  function showLayer(k) {\n"
        "    Object.keys(LAYERS).forEach(function (name) {\n"
        "      var lyr = LAYERS[name];\n"
        "      if (!lyr) return;\n"
        "      if (name === k) { if (!MAP.hasLayer(lyr)) lyr.addTo(MAP); }\n"
        "      else if (MAP.hasLayer(lyr)) { MAP.removeLayer(lyr); }\n"
        "    });\n"
        "    if (TT && MAP.hasLayer(TT) && TT.bringToFront) TT.bringToFront();\n"
        "    document.querySelectorAll('.wgtb-layer').forEach(function (b) {\n"
        "      b.classList.toggle('is-active', b.getAttribute('data-layer') === k);\n"
        "    });\n"
        "    if (body) body.innerHTML = LEGENDS[k] || '';\n"
        "  }\n"
        "  document.querySelectorAll('.wgtb-layer').forEach(function (b) {\n"
        "    b.addEventListener('click', function () { showLayer(b.getAttribute('data-layer')); });\n"
        "  });\n"
        "  var zi = document.querySelector('.wgtb-zoom-in');\n"
        "  var zo = document.querySelector('.wgtb-zoom-out');\n"
        "  if (zi) zi.addEventListener('click', function () { MAP.zoomIn(); });\n"
        "  if (zo) zo.addEventListener('click', function () { MAP.zoomOut(); });\n"
        "  // Pen the view to the GTA. The zoom-out floor is 'all of Toronto just fits\n"
        "  // the viewport' -- recomputed from the actual data bounds on load + resize\n"
        "  // (getBoundsZoom), so a phone floors lower than a wide desktop and neither\n"
        "  // can shrink the map past the coloured extent. maxBounds blocks panning it\n"
        "  // off-screen.\n"
        f"  var TO_BOUNDS = L.latLngBounds([[{data_bounds[1]}, {data_bounds[0]}], "
        f"[{data_bounds[3]}, {data_bounds[2]}]]);\n"
        "  MAP.options.maxBoundsViscosity = 0.5;\n"
        "  MAP.setMaxBounds([[43.25, -80.45], [44.20, -78.20]]);\n"
        "  function clampMinZoom() {\n"
        "    MAP.invalidateSize();\n"
        "    var z = MAP.getBoundsZoom(TO_BOUNDS, false);\n"
        "    MAP.setMinZoom(Math.max(10, Math.min(z, 12)));\n"
        "  }\n"
        "  showLayer('gap');\n"
        "  setTimeout(clampMinZoom, 200);\n"
        "  window.addEventListener('resize', clampMinZoom);\n"
        "})();\n"
        "</script>\n"
    )
    return scaffold, wiring


def _swatch_bar(colors: list[str]) -> str:
    segs = "".join(f'<span style="background:{c}"></span>' for c in colors)
    return f'<div class="wgtb-legend-bar">{segs}</div>'


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

    data_bounds = tuple(round(float(v), 4) for v in gdf.total_bounds)  # (minlon, minlat, maxlon, maxlat), EPSG:4326

    gap_abs = float(np.abs(gdf["equity_gap"]).max())
    gap_min = float(gdf["equity_gap"].min())
    gap_max = float(gdf["equity_gap"].max())
    acc_max = float(gdf["access_score"].max())
    need_max = float(gdf["need_score"].max())

    # ---- base map ----
    # tiles=None + explicit control=False TileLayers: keeps the basemap out of any
    # layer group. zoom_control=False: folium's default topleft zoom is removed --
    # the custom .wgtb-zoom pill (bottom-right, inside the map panel) replaces it.
    # min_zoom=10 + a GTA maxBounds: the user can't shrink the map past "all of
    # Toronto + a GTA margin (Mississauga .. Oshawa)" in view, and can't pan off it.
    m = folium.Map(
        location=TORONTO_CENTER,
        zoom_start=DEFAULT_ZOOM,
        min_zoom=10,
        tiles=None,
        zoom_control=False,
        attributionControl=False,   # all credits live in the custom footer
    )
    folium.TileLayer(
        tiles=ESRI_BASE_URL, attr=ESRI_ATTR, name="basemap",
        control=False, max_zoom=16,
    ).add_to(m)

    # ---- colour scales (branca colormaps, used ONLY to colour the polygons) ----
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
                "color": "#6b6486",
                "weight": 0.5,
                "fillOpacity": 0.78,
            }
        return style_function

    # ---- three choropleth layers ----
    # Hand-rolled folium.GeoJson + branca colormap (not folium.Choropleth: its d3
    # legend misbehaves with several on one map and will not centre a diverging
    # ramp on 0). control=False + show only the default: the custom Layers panel
    # adds/removes these; folium's LayerControl is not used at all.
    layers = [
        ("gap", "equity_gap", gap_cmap, True),
        ("access", "access_score", acc_cmap, False),
        ("need", "need_score", need_cmap, False),
    ]
    layer_js: dict[str, str] = {}
    for key, col, cmap, show in layers:
        gj = folium.GeoJson(
            gdf.__geo_interface__,
            name=key,
            control=False,
            show=show,
            style_function=_style(cmap, col),
            highlight_function=lambda _f: {"weight": 2, "color": C_ACTIVE_BORDER,
                                           "fillOpacity": 0.9},
            smooth_factor=0.5,
        )
        gj.add_to(m)
        layer_js[key] = gj.get_name()

    # Esri place-name labels. Leaflet keeps all tile layers in tilePane (below the
    # SVG overlayPane), so these sit *under* the choropleth, but the fill opacity
    # lets major labels (Toronto, North York, Scarborough, ...) read through.
    folium.TileLayer(
        tiles=ESRI_REF_URL, attr=ESRI_ATTR, name="labels",
        control=False, max_zoom=16, overlay=True,
    ).add_to(m)

    # ---- one always-on transparent tooltip layer on top (UNCHANGED) ----
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
            "background-color:#ffffff; color:#222; font-family:'Noto Sans', sans-serif; "
            "font-size:12px; padding:8px 10px; border:1px solid #bbb; border-radius:4px; "
            "box-shadow:0 1px 4px rgba(0,0,0,0.25);"
        ),
    )
    tt_gj = folium.GeoJson(
        gdf.__geo_interface__,
        name="Neighbourhood details (hover)",
        control=False,          # always on, not a toggle
        style_function=lambda _f: {"fillColor": "#000000", "color": "#000000",
                                   "weight": 0, "fillOpacity": 0},
        highlight_function=lambda _f: {"weight": 2.5, "color": C_ACTIVE_BORDER,
                                       "fillOpacity": 0.06},
        tooltip=tooltip,
        smooth_factor=0.5,
    )
    tt_gj.add_to(m)

    # ---- legend fragments (one per layer; the custom JS swaps them) ----
    legends = {
        "gap": (
            _swatch_bar(GAP_COLORS)
            + '<div class="wgtb-legend-ends">'
            f'<span>{gap_min:+.2f}</span><span>0</span><span>{gap_max:+.2f}</span></div>'
            f'<div class="wgtb-legend-key"><i style="background:{GAP_KEY_BLUE}"></i>'
            '<span>Blue: access exceeds need</span></div>'
            f'<div class="wgtb-legend-key"><i style="background:{GAP_KEY_PINK}"></i>'
            '<span>Pink: underserved relative to need</span></div>'
        ),
        "access": (
            _swatch_bar(YLORRD)
            + '<div class="wgtb-legend-ends">'
            f'<span>0.00</span><span>{acc_max:.2f}</span></div>'
            '<div class="wgtb-legend-note">Stop density + capacity-weighted peak '
            'frequency (incl. 500 m rapid-transit walkshed). Higher = better served.</div>'
        ),
        "need": (
            _swatch_bar(YLORRD)
            + '<div class="wgtb-legend-ends">'
            f'<span>0.00</span><span>{need_max:.2f}</span></div>'
            '<div class="wgtb-legend-note">Mean of low-income share, non-car-commute '
            'share, population density. Higher = more transit-dependent.</div>'
        ),
    }

    # ---- render, then wrap folium's output in the dark-UI shell ----
    os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
    html = m.get_root().render()

    map_name = m.get_name()
    scaffold, wiring = build_shell(
        map_name, layer_js, tt_gj.get_name(), legends, LICENCE_HTML, data_bounds
    )

    # 1. font + shell CSS into <head> (last, so it wins over folium's rules)
    head_inject = f'<link rel="stylesheet" href="{GOOGLE_FONT_URL}">\n{SHELL_CSS}\n'
    html = html.replace("</head>", head_inject + "</head>", 1)

    # 2. replace the bare folium-map div with the scaffold that embeds it
    html, n = re.subn(r'<div class="folium-map" id="[^"]+"\s*></div>', scaffold, html, count=1)
    assert n == 1, f"expected exactly one folium-map div, replaced {n}"

    # 3. wiring JS after folium's trailing <script>
    html = html.replace("</html>", wiring + "</html>", 1)

    # 4. vendor the Google font, then every remaining CDN asset
    html = vendor_google_font(html)
    html = inline_web_assets(html)

    with open(OUT_HTML, "w", encoding="utf-8") as fh:
        fh.write(html)

    remaining = re.findall(
        r'(?:src|href)="(https?://(?![^"]*basemaps|[^"]*tile|[^"]*leafletjs\.com)[^"]+)"', html
    )
    if remaining:
        print(f"  NOTE {len(remaining)} external ref(s) still linked: {sorted(set(remaining))}")

    size_kb = os.path.getsize(OUT_HTML) / 1024
    print(f"wrote {OUT_HTML}  ({size_kb:,.0f} KB)")
    print(f"  layers        : Equity gap / Access / Need (custom control, 'Equity gap' default)")
    print(f"  gap bins      : {gap_bins}  (symmetric, centred 0; |max gap| = {gap_abs:.3f})")
    print(f"  legend ends   : gap {gap_min:+.2f}..{gap_max:+.2f} · access 0..{acc_max:.2f} · need 0..{need_max:.2f}")
    print(f"  basemap       : Esri Dark Gray Canvas (keyless)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
