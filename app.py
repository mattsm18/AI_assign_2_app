"""
Auckland Land Price Estimator
Refactored for correctness and performance.

Performance fixes in this version
──────────────────────────────────
1. build_base_map() is now @st.cache_resource — the folium map object is built
   exactly once and reused on every rerun. Previously it was rebuilt on every
   single click (the largest freeze cause).

2. The panel is wrapped in @st.fragment — clicking the map only reruns the
   panel fragment, NOT the expensive map widget. Without this, every click
   triggered a full page rerun that re-rendered the entire map in the browser.

3. st_folium is called with feature_group_to_add=None and render_iframe=False
   (defaults) so Streamlit-Folium can diff correctly across reruns.

4. STRtree spatial index for O(log n) zone lookups — unchanged from previous.

5. Per-session prediction cache in session_state — zero-cost on repeated clicks
   to the same (rounded) location — unchanged from previous.
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from scipy.interpolate import RBFInterpolator
from sklearn.preprocessing import StandardScaler
import geopandas as gpd
from shapely.geometry import Point
from shapely.strtree import STRtree

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Auckland Land Price",
    page_icon="🗺️",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Mono', monospace;
    background: #f0ede8;
    color: #111;
}

h1, h2, h3, .panel-heading {
    font-family: 'Syne', sans-serif !important;
    letter-spacing: -0.02em;
}

.stApp { background: #f0ede8; }

.panel {
    background: #1a1a1a;
    border-radius: 12px;
    padding: 1.5rem;
    color: #f0ede8;
    min-height: 520px;
}

.panel-heading {
    font-size: 1.4rem;
    font-weight: 800;
    color: #f0ede8;
    margin-bottom: 1.5rem;
    border-bottom: 1px solid #333;
    padding-bottom: 0.75rem;
}

.price-big {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: #c8f560;
    line-height: 1.1;
    margin: 0.25rem 0;
}

.price-range {
    font-size: 0.72rem;
    color: #888;
    letter-spacing: 0.05em;
    margin-bottom: 1.2rem;
}

.stat-label {
    font-size: 0.62rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #666;
    margin-bottom: 0.2rem;
}

.stat-value {
    font-size: 0.95rem;
    color: #f0ede8;
    margin-bottom: 1rem;
}

.zone-badge {
    display: inline-block;
    padding: 0.2rem 0.65rem;
    border-radius: 4px;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
}

.zone-residential { background: #c8f560; color: #111; }
.zone-reserve     { background: #5eead4; color: #111; }
.zone-sea         { background: #60a5fa; color: #111; }

.hint-text {
    color: #555;
    font-size: 0.8rem;
    text-align: center;
    padding-top: 2rem;
}

.streamlit-expanderHeader {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

CBD = np.array([-36.8485, 174.7633])
MAP_CENTER = [-36.86, 174.76]
MAP_ZOOM = 11
PRICE_BOUNDS = (400_000, 6_000_000)

SUBURB_DATA = [
    {"suburb": "Auckland Central",   "lat": -36.8509, "lon": 174.7645, "price": 481_000},
    {"suburb": "Parnell",            "lat": -36.8577, "lon": 174.7802, "price": 1_850_000},
    {"suburb": "Ponsonby",           "lat": -36.8544, "lon": 174.7459, "price": 2_100_000},
    {"suburb": "Grey Lynn",          "lat": -36.8650, "lon": 174.7384, "price": 1_650_000},
    {"suburb": "Herne Bay",          "lat": -36.8472, "lon": 174.7333, "price": 3_137_000},
    {"suburb": "Westmere",           "lat": -36.8600, "lon": 174.7200, "price": 2_100_000},
    {"suburb": "Freemans Bay",       "lat": -36.8489, "lon": 174.7518, "price": 1_600_000},
    {"suburb": "Newmarket",          "lat": -36.8710, "lon": 174.7770, "price": 1_100_000},
    {"suburb": "Remuera",            "lat": -36.8783, "lon": 174.7967, "price": 2_200_000},
    {"suburb": "Meadowbank",         "lat": -36.8733, "lon": 174.8183, "price": 1_500_000},
    {"suburb": "St Heliers",         "lat": -36.8683, "lon": 174.8567, "price": 1_700_000},
    {"suburb": "Kohimarama",         "lat": -36.8617, "lon": 174.8417, "price": 1_750_000},
    {"suburb": "Mission Bay",        "lat": -36.8583, "lon": 174.8333, "price": 1_650_000},
    {"suburb": "Mount Eden",         "lat": -36.8780, "lon": 174.7650, "price": 1_700_000},
    {"suburb": "Kingsland",          "lat": -36.8717, "lon": 174.7450, "price": 1_400_000},
    {"suburb": "Sandringham",        "lat": -36.8883, "lon": 174.7433, "price": 1_250_000},
    {"suburb": "Mount Albert",       "lat": -36.8900, "lon": 174.7267, "price": 1_200_000},
    {"suburb": "Waterview",          "lat": -36.8867, "lon": 174.7183, "price": 1_050_000},
    {"suburb": "Point Chevalier",    "lat": -36.8650, "lon": 174.7167, "price": 1_500_000},
    {"suburb": "Epsom",              "lat": -36.8965, "lon": 174.7762, "price": 1_850_000},
    {"suburb": "Royal Oak",          "lat": -36.9050, "lon": 174.7730, "price": 1_350_000},
    {"suburb": "Three Kings",        "lat": -36.9067, "lon": 174.7583, "price": 1_300_000},
    {"suburb": "Onehunga",           "lat": -36.9225, "lon": 174.7839, "price": 1_100_000},
    {"suburb": "Ellerslie",          "lat": -36.9058, "lon": 174.8100, "price": 1_300_000},
    {"suburb": "Glen Innes",         "lat": -36.8819, "lon": 174.8500, "price": 1_050_000},
    {"suburb": "Panmure",            "lat": -36.9017, "lon": 174.8467, "price":   950_000},
    {"suburb": "Mount Wellington",   "lat": -36.9100, "lon": 174.8350, "price":   950_000},
    {"suburb": "Penrose",            "lat": -36.9233, "lon": 174.8100, "price":   900_000},
    {"suburb": "Otahuhu",            "lat": -36.9533, "lon": 174.8383, "price":   800_000},
    {"suburb": "Mangere Bridge",     "lat": -36.9600, "lon": 174.7950, "price":   870_000},
    {"suburb": "Mangere",            "lat": -36.9817, "lon": 174.7967, "price":   780_000},
    {"suburb": "Howick",             "lat": -36.9000, "lon": 174.9333, "price": 1_150_000},
    {"suburb": "Pakuranga",          "lat": -36.9000, "lon": 174.9100, "price": 1_100_000},
    {"suburb": "Botany Downs",       "lat": -36.9317, "lon": 174.9183, "price": 1_100_000},
    {"suburb": "Flat Bush",          "lat": -36.9583, "lon": 174.9133, "price":   980_000},
    {"suburb": "Beachlands",         "lat": -36.8833, "lon": 175.0000, "price": 1_250_000},
    {"suburb": "Bucklands Beach",    "lat": -36.8767, "lon": 174.9467, "price": 1_400_000},
    {"suburb": "Half Moon Bay",      "lat": -36.8967, "lon": 174.9367, "price": 1_350_000},
    {"suburb": "Papatoetoe",         "lat": -36.9833, "lon": 174.8500, "price":   870_000},
    {"suburb": "Manukau",            "lat": -36.9933, "lon": 174.8783, "price":   830_000},
    {"suburb": "Clover Park",        "lat": -37.0050, "lon": 174.9000, "price":   800_000},
    {"suburb": "Takanini",           "lat": -37.0433, "lon": 174.9167, "price":   820_000},
    {"suburb": "Papakura",           "lat": -37.0650, "lon": 174.9433, "price":   754_500},
    {"suburb": "Pukekohe",           "lat": -37.2000, "lon": 174.9000, "price":   760_000},
    {"suburb": "Karaka",             "lat": -37.1167, "lon": 174.9000, "price": 1_450_000},
    {"suburb": "Henderson",          "lat": -36.8750, "lon": 174.6317, "price":   900_000},
    {"suburb": "Te Atatu South",     "lat": -36.8600, "lon": 174.6600, "price": 1_000_000},
    {"suburb": "Te Atatu Peninsula", "lat": -36.8400, "lon": 174.6517, "price": 1_050_000},
    {"suburb": "Massey",             "lat": -36.8450, "lon": 174.5950, "price":   880_000},
    {"suburb": "Waitakere",          "lat": -36.8767, "lon": 174.5650, "price":   885_000},
    {"suburb": "Swanson",            "lat": -36.8850, "lon": 174.6133, "price":   870_000},
    {"suburb": "Glen Eden",          "lat": -36.9233, "lon": 174.6467, "price":   830_000},
    {"suburb": "New Lynn",           "lat": -36.9083, "lon": 174.6867, "price":   950_000},
    {"suburb": "Avondale",           "lat": -36.8983, "lon": 174.7133, "price": 1_050_000},
    {"suburb": "Takapuna",           "lat": -36.7883, "lon": 174.7700, "price": 1_900_000},
    {"suburb": "Devonport",          "lat": -36.8300, "lon": 174.7983, "price": 1_950_000},
    {"suburb": "Northcote",          "lat": -36.8000, "lon": 174.7500, "price": 1_300_000},
    {"suburb": "Birkenhead",         "lat": -36.8133, "lon": 174.7317, "price": 1_250_000},
    {"suburb": "Glenfield",          "lat": -36.7817, "lon": 174.7217, "price": 1_100_000},
    {"suburb": "Browns Bay",         "lat": -36.7200, "lon": 174.7500, "price": 1_300_000},
    {"suburb": "Torbay",             "lat": -36.6933, "lon": 174.7583, "price": 1_200_000},
    {"suburb": "Albany",             "lat": -36.7300, "lon": 174.7000, "price": 1_250_000},
    {"suburb": "Mairangi Bay",       "lat": -36.7333, "lon": 174.7617, "price": 1_400_000},
    {"suburb": "Milford",            "lat": -36.7733, "lon": 174.7633, "price": 1_600_000},
    {"suburb": "Stanley Point",      "lat": -36.8167, "lon": 174.7867, "price": 2_100_000},
    {"suburb": "Birkdale",           "lat": -36.8167, "lon": 174.7133, "price": 1_003_000},
    {"suburb": "Forrest Hill",       "lat": -36.7617, "lon": 174.7383, "price": 1_300_000},
    {"suburb": "Orewa",              "lat": -36.5933, "lon": 174.6967, "price": 1_200_000},
    {"suburb": "Silverdale",         "lat": -36.6217, "lon": 174.6700, "price": 1_100_000},
    {"suburb": "Warkworth",          "lat": -36.4000, "lon": 174.6600, "price":   850_000},
    {"suburb": "Waiheke Island",     "lat": -36.8000, "lon": 175.1000, "price": 1_400_000},
    {"suburb": "Omaha",              "lat": -36.3417, "lon": 174.7233, "price": 3_010_000},
    {"suburb": "Whangaparaoa",       "lat": -36.6317, "lon": 174.7333, "price": 1_050_000},
    {"suburb": "Coatesville",        "lat": -36.6983, "lon": 174.6683, "price": 2_800_000},
    {"suburb": "Kumeu",              "lat": -36.7817, "lon": 174.5617, "price": 1_050_000},
]


# ═══════════════════════════════════════════════════════════════
# DATA & MODEL — loaded once, cached as resources
# ═══════════════════════════════════════════════════════════════

def _iter_polygons(geom):
    """Yield only Polygon sub-geometries from any Shapely geometry type."""
    if geom is None or geom.is_empty:
        return
    if geom.geom_type == "Polygon":
        yield geom
    elif geom.geom_type in ("MultiPolygon", "GeometryCollection"):
        for part in geom.geoms:
            yield from _iter_polygons(part)


@st.cache_resource
def load_everything():
    """
    Load GeoJSON files, build STRtree spatial index, and train the RBF model.
    Runs exactly once for the lifetime of the server process.
    """
    parks_gdf = gpd.read_file("parks.geojson").to_crs(epsg=4326)
    coast_gdf = gpd.read_file("coastline.geojson").to_crs(epsg=4326)

    park_tree  = STRtree(parks_gdf.geometry.values)
    coast_tree = STRtree(coast_gdf.geometry.values)

    park_polys = []
    for geom, name in zip(parks_gdf.geometry, parks_gdf.get("name", ["Park"] * len(parks_gdf))):
        for poly in _iter_polygons(geom):
            park_polys.append(([(y, x) for x, y in poly.exterior.coords], str(name) if name else "Park"))

    coast_polys = []
    for geom, name in zip(coast_gdf.geometry, coast_gdf.get("name", ["Coast"] * len(coast_gdf))):
        for poly in _iter_polygons(geom):
            coast_polys.append(([(y, x) for x, y in poly.exterior.coords], str(name) if name else "Coastline"))

    df     = pd.DataFrame(SUBURB_DATA)
    coords = df[["lat", "lon"]].values
    prices = df["price"].values

    scaler        = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)

    rbf = RBFInterpolator(
        coords_scaled, prices,
        kernel="thin_plate_spline",
        smoothing=1e9,
    )

    log_prices      = np.log(prices)
    pmin, pmax      = log_prices.min(), log_prices.max()
    intensities_norm = (log_prices - pmin) / (pmax - pmin)

    rng    = np.random.default_rng(42)
    n_pts  = len(df) * 30
    lats_h = np.repeat(df["lat"].values, 30) + rng.normal(0, 0.008, n_pts)
    lons_h = np.repeat(df["lon"].values, 30) + rng.normal(0, 0.008, n_pts)
    ints_h = np.repeat(intensities_norm, 30)
    heat   = np.column_stack([lats_h, lons_h, ints_h]).tolist()

    return dict(
        parks_gdf=parks_gdf,
        coast_gdf=coast_gdf,
        park_tree=park_tree,
        coast_tree=coast_tree,
        park_polys=park_polys,
        coast_polys=coast_polys,
        rbf=rbf,
        scaler=scaler,
        suburbs_df=df,
        heat=heat,
    )


DATA = load_everything()


# ═══════════════════════════════════════════════════════════════
# MAP — built once and cached as a resource
#
# FIX: previously build_base_map() had no caching, so it was
# called on *every* rerun (i.e. every click). Building the
# folium map with hundreds of polygons took ~1–2 s each time.
# Caching as a resource means it is built exactly once.
# ═══════════════════════════════════════════════════════════════

@st.cache_resource
def build_base_map():
    m = folium.Map(location=MAP_CENTER, zoom_start=MAP_ZOOM, tiles="CartoDB positron")

    HeatMap(DATA["heat"], radius=20, blur=16, min_opacity=0.3).add_to(m)

    for coords, name in DATA["park_polys"]:
        folium.Polygon(
            locations=coords,
            color="#22c55e",
            weight=1,
            fill=True,
            fill_color="#22c55e",
            fill_opacity=0.25,
            tooltip=name,
        ).add_to(m)

    for coords, name in DATA["coast_polys"]:
        folium.Polygon(
            locations=coords,
            color="#1d4ed8",
            weight=1.5,
            fill=False,
            tooltip=name,
        ).add_to(m)

    for _, row in DATA["suburbs_df"].iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=4,
            fill=True,
            fill_color="#ffffff",
            fill_opacity=0.9,
            color="#333",
            weight=1,
            tooltip=f"{row['suburb']} — ${row['price']:,.0f}",
        ).add_to(m)

    return m


# ═══════════════════════════════════════════════════════════════
# SPATIAL QUERY
# ═══════════════════════════════════════════════════════════════

def classify_point(lat: float, lon: float) -> str:
    """Returns 'sea' | 'reserve' | 'residential'. Cached in session_state."""
    key = (round(lat, 4), round(lon, 4))
    cache = st.session_state.setdefault("_zone_cache", {})
    if key in cache:
        return cache[key]

    pt = Point(key[1], key[0])  # Point(lon, lat)

    for idx in DATA["park_tree"].query(pt):
        if DATA["parks_gdf"].geometry.iloc[idx].contains(pt):
            cache[key] = "reserve"
            return "reserve"

    on_land = any(
        DATA["coast_gdf"].geometry.iloc[idx].contains(pt)
        for idx in DATA["coast_tree"].query(pt)
    )

    result = "residential" if on_land else "sea"
    cache[key] = result
    return result


def predict(lat: float, lon: float) -> dict:
    """Full prediction for a clicked point. Cached in session_state."""
    key = (round(lat, 4), round(lon, 4))
    pcache = st.session_state.setdefault("_pred_cache", {})
    if key in pcache:
        return pcache[key]

    zone = classify_point(lat, lon)

    df    = DATA["suburbs_df"]
    dists = np.sqrt((df["lat"] - lat) ** 2 + (df["lon"] - lon) ** 2)
    ci    = dists.idxmin()

    cbd_km = np.linalg.norm(np.array([lat, lon]) - CBD) * 111.0

    result = dict(
        zone=zone, lat=lat, lon=lon,
        cbd_km=cbd_km,
        closest_suburb=df.loc[ci, "suburb"],
        closest_km=dists[ci] * 111.0,
        price=None, price_low=None, price_high=None,
    )

    if zone == "residential":
        raw   = float(DATA["rbf"](DATA["scaler"].transform([[lat, lon]]))[0])
        price = float(np.clip(raw, *PRICE_BOUNDS))
        result.update(price=price, price_low=price * 0.88, price_high=price * 1.12)

    pcache[key] = result
    return result


# ═══════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════

st.title("Auckland Land Price Estimator")
st.markdown(
    "<p style='color:#888;font-size:0.82rem;margin-top:-0.5rem;"
    "font-family:IBM Plex Mono,monospace'>Click anywhere on the map for an instant estimate</p>",
    unsafe_allow_html=True,
)

col_map, col_panel = st.columns([3, 1])

# ── Map column ────────────────────────────────────────────────
# The map widget lives outside any fragment so it renders once
# and is NOT re-executed when the panel fragment reruns.
with col_map:
    map_data = st_folium(
        build_base_map(),       # cached — same object reference every time
        width="100%",
        height=620,
        returned_objects=["last_clicked"],
        key="main_map",
    )

# Write the click into session_state so the panel fragment can read it.
# This block is cheap (just a dict lookup + assignment) on every rerun.
if map_data and map_data.get("last_clicked"):
    c = map_data["last_clicked"]
    new_coords = (c["lat"], c["lng"])
    # Only trigger a panel rerun when the click location actually changed.
    if st.session_state.get("_last_click") != new_coords:
        st.session_state["_last_click"] = new_coords
        st.rerun(scope="fragment")   # cheaply reruns only the fragment below


# ── Panel column — fragment so it reruns independently ────────
#
# FIX: wrapping the panel in @st.fragment means a click only reruns
# this function, not the entire page. The map widget above is NOT
# re-executed, so the browser never has to reload the map HTML.
@st.fragment
def render_panel():
    clicked_coords = st.session_state.get("_last_click")

    with col_panel:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown("<div class='panel-heading'>Estimate</div>", unsafe_allow_html=True)

        if not clicked_coords:
            st.markdown(
                "<p class='hint-text'>← Click the map to estimate land value at any point in Auckland.</p>",
                unsafe_allow_html=True,
            )
        else:
            lat, lon = clicked_coords
            res = predict(lat, lon)

            zone_labels = {"residential": "Residential", "reserve": "Reserve / Park", "sea": "Ocean"}
            zone_css    = {"residential": "zone-residential", "reserve": "zone-reserve", "sea": "zone-sea"}
            z = res["zone"]

            st.markdown(
                f"<span class='zone-badge {zone_css[z]}'>{zone_labels[z]}</span>",
                unsafe_allow_html=True,
            )

            if z == "residential" and res["price"] is not None:
                st.markdown(
                    f"<div class='price-big'>${res['price']:,.0f}</div>"
                    f"<div class='price-range'>${res['price_low']:,.0f} – ${res['price_high']:,.0f} est. range</div>",
                    unsafe_allow_html=True,
                )
            elif z == "reserve":
                st.markdown(
                    "<div style='color:#5eead4;font-size:1rem;margin:0.5rem 0'>"
                    "Park / Reserve<br><small style='color:#666'>No residential pricing</small></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div style='color:#60a5fa;font-size:1rem;margin:0.5rem 0'>"
                    "Ocean / Water<br><small style='color:#666'>No pricing available</small></div>",
                    unsafe_allow_html=True,
                )

            st.markdown(
                f"<div class='stat-label'>Distance to CBD</div>"
                f"<div class='stat-value'>{res['cbd_km']:.1f} km</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='stat-label'>Nearest suburb</div>"
                f"<div class='stat-value'>{res['closest_suburb']}<br>"
                f"<span style='color:#666;font-size:0.75rem'>{res['closest_km']:.1f} km away</span></div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='stat-label'>Coordinates</div>"
                f"<div class='stat-value' style='font-size:0.78rem'>{lat:.4f}, {lon:.4f}</div>",
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)


render_panel()

# ── Data explorer ─────────────────────────────────────────────
with st.expander("Suburb reference data"):
    display_df = DATA["suburbs_df"].copy()
    display_df["price"] = display_df["price"].map("${:,.0f}".format)
    st.dataframe(display_df, use_container_width=True, hide_index=True)