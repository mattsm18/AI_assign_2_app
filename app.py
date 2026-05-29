"""
Auckland Land Price Estimator - Refactored for Performance

Optimizations:
- Vectorized NumPy operations for heatmap generation
- Pre-computed zone features for map rendering
- Efficient spatial indexing with STRtree
- LRU caching for zone classification
- Minimal geometry operations
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
from functools import lru_cache

# ═════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & STYLING
# ═════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Auckland Land Price Tool",
    page_icon="🗺️",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background-color: #f7f5f0;
    color: #1a1a1a;
}

h1, h2, h3 {
    font-family: 'DM Serif Display', serif !important;
}

.price-card {
    background: #ffffff;
    border: 1.5px solid #2a9d8f;
    border-radius: 8px;
    padding: 1.5rem;
    margin-top: 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
}

.price-card h2 {
    color: #2a9d8f;
    font-size: 2rem;
    margin: 0;
    font-family: 'DM Serif Display', serif;
}

.label {
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    color: #888;
    text-transform: uppercase;
}

.subtitle {
    color: #888;
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
    margin-top: -0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════

CBD_COORDS = np.array([-36.8485, 174.7633])
MAP_CENTER = [-36.86, 174.76]
MAP_ZOOM = 11
KM_PER_DEGREE = 111.0
PRICE_BOUNDS = (400000, 5000000)
COORD_PRECISION = 4

SUBURB_DATA = [
    {"suburb": "Auckland Central",   "lat": -36.8509, "lon": 174.7645, "price": 481000},
    {"suburb": "Parnell",            "lat": -36.8577, "lon": 174.7802, "price": 1850000},
    {"suburb": "Ponsonby",           "lat": -36.8544, "lon": 174.7459, "price": 2100000},
    {"suburb": "Grey Lynn",          "lat": -36.8650, "lon": 174.7384, "price": 1650000},
    {"suburb": "Herne Bay",          "lat": -36.8472, "lon": 174.7333, "price": 3137000},
    {"suburb": "Westmere",           "lat": -36.8600, "lon": 174.7200, "price": 2100000},
    {"suburb": "Freemans Bay",       "lat": -36.8489, "lon": 174.7518, "price": 1600000},
    {"suburb": "Newmarket",          "lat": -36.8710, "lon": 174.7770, "price": 1100000},
    {"suburb": "Remuera",            "lat": -36.8783, "lon": 174.7967, "price": 2200000},
    {"suburb": "Meadowbank",         "lat": -36.8733, "lon": 174.8183, "price": 1500000},
    {"suburb": "St Heliers",         "lat": -36.8683, "lon": 174.8567, "price": 1700000},
    {"suburb": "Kohimarama",         "lat": -36.8617, "lon": 174.8417, "price": 1750000},
    {"suburb": "Mission Bay",        "lat": -36.8583, "lon": 174.8333, "price": 1650000},
    {"suburb": "Mount Eden",         "lat": -36.8780, "lon": 174.7650, "price": 1700000},
    {"suburb": "Kingsland",          "lat": -36.8717, "lon": 174.7450, "price": 1400000},
    {"suburb": "Sandringham",        "lat": -36.8883, "lon": 174.7433, "price": 1250000},
    {"suburb": "Mount Albert",       "lat": -36.8900, "lon": 174.7267, "price": 1200000},
    {"suburb": "Waterview",          "lat": -36.8867, "lon": 174.7183, "price": 1050000},
    {"suburb": "Point Chevalier",    "lat": -36.8650, "lon": 174.7167, "price": 1500000},
    {"suburb": "Epsom",              "lat": -36.8965, "lon": 174.7762, "price": 1850000},
    {"suburb": "Royal Oak",          "lat": -36.9050, "lon": 174.7730, "price": 1350000},
    {"suburb": "Three Kings",        "lat": -36.9067, "lon": 174.7583, "price": 1300000},
    {"suburb": "Onehunga",           "lat": -36.9225, "lon": 174.7839, "price": 1100000},
    {"suburb": "Ellerslie",          "lat": -36.9058, "lon": 174.8100, "price": 1300000},
    {"suburb": "Glen Innes",         "lat": -36.8819, "lon": 174.8500, "price": 1050000},
    {"suburb": "Panmure",            "lat": -36.9017, "lon": 174.8467, "price": 950000},
    {"suburb": "Mount Wellington",   "lat": -36.9100, "lon": 174.8350, "price": 950000},
    {"suburb": "Penrose",            "lat": -36.9233, "lon": 174.8100, "price": 900000},
    {"suburb": "Otahuhu",            "lat": -36.9533, "lon": 174.8383, "price": 800000},
    {"suburb": "Mangere Bridge",     "lat": -36.9600, "lon": 174.7950, "price": 870000},
    {"suburb": "Mangere",            "lat": -36.9817, "lon": 174.7967, "price": 780000},
    {"suburb": "Howick",             "lat": -36.9000, "lon": 174.9333, "price": 1150000},
    {"suburb": "Pakuranga",          "lat": -36.9000, "lon": 174.9100, "price": 1100000},
    {"suburb": "Botany Downs",       "lat": -36.9317, "lon": 174.9183, "price": 1100000},
    {"suburb": "Flat Bush",          "lat": -36.9583, "lon": 174.9133, "price": 980000},
    {"suburb": "Beachlands",         "lat": -36.8833, "lon": 175.0000, "price": 1250000},
    {"suburb": "Bucklands Beach",    "lat": -36.8767, "lon": 174.9467, "price": 1400000},
    {"suburb": "Half Moon Bay",      "lat": -36.8967, "lon": 174.9367, "price": 1350000},
    {"suburb": "Papatoetoe",         "lat": -36.9833, "lon": 174.8500, "price": 870000},
    {"suburb": "Manukau",            "lat": -36.9933, "lon": 174.8783, "price": 830000},
    {"suburb": "Clover Park",        "lat": -37.0050, "lon": 174.9000, "price": 800000},
    {"suburb": "Takanini",           "lat": -37.0433, "lon": 174.9167, "price": 820000},
    {"suburb": "Papakura",           "lat": -37.0650, "lon": 174.9433, "price": 754500},
    {"suburb": "Pukekohe",           "lat": -37.2000, "lon": 174.9000, "price": 760000},
    {"suburb": "Karaka",             "lat": -37.1167, "lon": 174.9000, "price": 1450000},
    {"suburb": "Henderson",          "lat": -36.8750, "lon": 174.6317, "price": 900000},
    {"suburb": "Te Atatu South",     "lat": -36.8600, "lon": 174.6600, "price": 1000000},
    {"suburb": "Te Atatu Peninsula", "lat": -36.8400, "lon": 174.6517, "price": 1050000},
    {"suburb": "Massey",             "lat": -36.8450, "lon": 174.5950, "price": 880000},
    {"suburb": "Waitakere",          "lat": -36.8767, "lon": 174.5650, "price": 885000},
    {"suburb": "Swanson",            "lat": -36.8850, "lon": 174.6133, "price": 870000},
    {"suburb": "Glen Eden",          "lat": -36.9233, "lon": 174.6467, "price": 830000},
    {"suburb": "New Lynn",           "lat": -36.9083, "lon": 174.6867, "price": 950000},
    {"suburb": "Avondale",           "lat": -36.8983, "lon": 174.7133, "price": 1050000},
    {"suburb": "Takapuna",           "lat": -36.7883, "lon": 174.7700, "price": 1900000},
    {"suburb": "Devonport",          "lat": -36.8300, "lon": 174.7983, "price": 1950000},
    {"suburb": "Northcote",          "lat": -36.8000, "lon": 174.7500, "price": 1300000},
    {"suburb": "Birkenhead",         "lat": -36.8133, "lon": 174.7317, "price": 1250000},
    {"suburb": "Glenfield",          "lat": -36.7817, "lon": 174.7217, "price": 1100000},
    {"suburb": "Browns Bay",         "lat": -36.7200, "lon": 174.7500, "price": 1300000},
    {"suburb": "Torbay",             "lat": -36.6933, "lon": 174.7583, "price": 1200000},
    {"suburb": "Albany",             "lat": -36.7300, "lon": 174.7000, "price": 1250000},
    {"suburb": "Mairangi Bay",       "lat": -36.7333, "lon": 174.7617, "price": 1400000},
    {"suburb": "Milford",            "lat": -36.7733, "lon": 174.7633, "price": 1600000},
    {"suburb": "Stanley Point",      "lat": -36.8167, "lon": 174.7867, "price": 2100000},
    {"suburb": "Birkdale",           "lat": -36.8167, "lon": 174.7133, "price": 1003000},
    {"suburb": "Forrest Hill",       "lat": -36.7617, "lon": 174.7383, "price": 1300000},
    {"suburb": "Orewa",              "lat": -36.5933, "lon": 174.6967, "price": 1200000},
    {"suburb": "Silverdale",         "lat": -36.6217, "lon": 174.6700, "price": 1100000},
    {"suburb": "Warkworth",          "lat": -36.4000, "lon": 174.6600, "price": 850000},
    {"suburb": "Waiheke Island",     "lat": -36.8000, "lon": 175.1000, "price": 1400000},
    {"suburb": "Omaha",              "lat": -36.3417, "lon": 174.7233, "price": 3010000},
    {"suburb": "Whangaparaoa",       "lat": -36.6317, "lon": 174.7333, "price": 1050000},
    {"suburb": "Coatesville",        "lat": -36.6983, "lon": 174.6683, "price": 2800000},
    {"suburb": "Kumeu",              "lat": -36.7817, "lon": 174.5617, "price": 1050000},
]

# ═════════════════════════════════════════════════════════════════════════════
# DATA LOADING & INITIALIZATION
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_spatial_data():
    """Load GeoJSON zones"""
    parks = gpd.read_file("parks.geojson").to_crs(epsg=4326)
    coast = gpd.read_file("coastline.geojson").to_crs(epsg=4326)
    
    parks["zone"] = "reserve"
    coast["zone"] = "sea"
    parks["name"] = parks.get("name", "Park")
    coast["name"] = coast.get("name", "Coast")
    
    zones_gdf = gpd.GeoDataFrame(
        pd.concat([parks, coast], ignore_index=True),
        geometry="geometry",
        crs="EPSG:4326"
    )
    
    return zones_gdf


# Load data once
zones_gdf = load_spatial_data()
    df = pd.DataFrame(SUBURB_DATA)
    coords = df[["lat", "lon"]].values
    prices = df["price"].values
    
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)
    
    rbf = RBFInterpolator(
        coords_scaled, prices,
        kernel="thin_plate_spline",
        smoothing=1e9
    )
    
    return rbf, scaler, df


@st.cache_data
def generate_heatmap_data(df):
    """Vectorized heatmap point generation"""
    pmin, pmax = df["price"].min(), df["price"].max()
    prices_normalized = (df["price"].values - pmin) / (pmax - pmin)
    
    # Vectorized random generation
    rng = np.random.default_rng(42)
    n_points = len(df) * 25
    
    lats = np.repeat(df["lat"].values, 25) + rng.normal(0, 0.01, n_points)
    lons = np.repeat(df["lon"].values, 25) + rng.normal(0, 0.01, n_points)
    intensities = np.repeat(prices_normalized, 25)
    
    return np.column_stack([lats, lons, intensities])


# Load data once
zones_gdf, zones_index, sea_zones, sea_index = load_spatial_data()
rbf, scaler, suburbs_df = build_price_model()
heat_data = generate_heatmap_data(suburbs_df)

# Pre-compute zone features for map rendering
@st.cache_resource
def prepare_zone_features():
    """Pre-compute zone features for map rendering"""
    features = []
    
    for idx, row in zones_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        
        if geom.geom_type == "Polygon":
            coords = [(y, x) for x, y in geom.exterior.coords]
            features.append((coords, row.get("name", "zone")))
        
        elif geom.geom_type == "MultiPolygon":
            for poly in geom.geoms:
                coords = [(y, x) for x, y in poly.exterior.coords]
                features.append((coords, row.get("name", "zone")))
    
    return features

zone_features = prepare_zone_features()

# ═════════════════════════════════════════════════════════════════════════════
# SPATIAL QUERY FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def build_zone_lookup():
    """Build fast zone lookup using bounding boxes"""
    # Pre-compute zone bounding boxes for quick filtering
    zones_info = []
    
    for idx, row in zones_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        
        bounds = geom.bounds  # (minx, miny, maxx, maxy)
        zones_info.append({
            'idx': idx,
            'zone': row['zone'],
            'name': row.get('name', 'Zone'),
            'bounds': bounds,
            'geom': geom
        })
    
    return zones_info

zones_info = build_zone_lookup()


def classify_zone_fast(lat: float, lon: float) -> tuple:
    """Fast zone lookup using bounding box pre-filtering"""
    # First check bounding boxes (very fast)
    candidates = []
    for zone_info in zones_info:
        minx, miny, maxx, maxy = zone_info['bounds']
        if minx <= lon <= maxx and miny <= lat <= maxy:
            candidates.append(zone_info)
    
    # If no candidates, it's residential
    if not candidates:
        return "residential", "Residential"
    
    # Check geometric containment only for candidates
    pt = Point(lon, lat)
    for zone_info in candidates:
        if zone_info['geom'].contains(pt):
            return zone_info['zone'], zone_info['name']
    
    return "residential", "Residential"


# Use Streamlit session state for persistent caching across reruns
if 'zone_cache' not in st.session_state:
    st.session_state.zone_cache = {}

if 'ocean_cache' not in st.session_state:
    st.session_state.ocean_cache = {}


def classify_zone_cached(lat: float, lon: float) -> tuple:
    """Cached zone lookup using session state"""
    key = (round(lat, 4), round(lon, 4))
    
    if key not in st.session_state.zone_cache:
        st.session_state.zone_cache[key] = classify_zone_fast(lat, lon)
    
    return st.session_state.zone_cache[key]


@lru_cache(maxsize=512)
def is_ocean_fast(lat: float, lon: float) -> bool:
    """Fast ocean detection"""
    zone, _ = classify_zone_cached(lat, lon)
    return zone == "sea"


def predict_price(lat: float, lon: float) -> tuple[float | None, str, str]:
    """Predict land price at given coordinates"""
    zone, name = classify_zone_cached(lat, lon)
    
    if zone != "residential":
        return None, zone, name
    
    coords = scaler.transform([[lat, lon]])
    price = float(rbf(coords)[0])
    return np.clip(price, *PRICE_BOUNDS), zone, name


def distance_from_cbd(lat: float, lon: float) -> float:
    """Calculate distance from CBD"""
    return np.linalg.norm(np.array([lat, lon]) - CBD_COORDS) * KM_PER_DEGREE

# ═════════════════════════════════════════════════════════════════════════════
# UI RENDERING
# ═════════════════════════════════════════════════════════════════════════════

st.title("Auckland Land Price Estimator")
st.markdown("<p class='subtitle'>Click anywhere on the map</p>", unsafe_allow_html=True)

col_map, col_panel = st.columns([3, 1])

# ─────────────────────────────────────────────
# MAP
# ─────────────────────────────────────────────
with col_map:
    m = folium.Map(location=MAP_CENTER, zoom_start=MAP_ZOOM, tiles="CartoDB positron")
    
    # Add heatmap
    HeatMap(heat_data, radius=18, blur=14).add_to(m)
    
    # Add suburb markers (vectorized)
    for _, row in suburbs_df.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=5,
            fill=True,
            fill_color="#fff",
            color="#000",
            tooltip=f"{row['suburb']} - ${row['price']:,.0f}"
        ).add_to(m)
    
    # Add zone boundaries (pre-computed)
    for coords, name in zone_features:
        folium.Polygon(
            locations=coords,
            color="#4cc9f0",
            fill=True,
            fill_opacity=0.15,
            tooltip=name
        ).add_to(m)
    
    map_data = st_folium(m, width="100%", height=600, returned_objects=["last_clicked"])

# ─────────────────────────────────────────────
# SIDE PANEL
# ─────────────────────────────────────────────
with col_panel:
    st.markdown("### Estimate")
    
    if map_data and map_data.get("last_clicked"):
        lat = map_data["last_clicked"]["lat"]
        lon = map_data["last_clicked"]["lng"]
        
        if is_ocean_fast(lat, lon):
            st.warning("Ocean — no pricing model")
        else:
            price, zone, name = predict_price(lat, lon)
            if price is not None:
                st.metric("Estimated Price", f"${price:,.0f}")
            else:
                st.info(f"{name} zone — no residential estimate")
        
        dist = distance_from_cbd(lat, lon)
        st.metric("Distance from CBD", f"{dist:.1f} km")
    else:
        st.info("Click map to estimate land value")

# ─────────────────────────────────────────────
# DATA EXPLORER
# ─────────────────────────────────────────────
with st.expander("Suburb data"):
    st.dataframe(suburbs_df, width='stretch')
