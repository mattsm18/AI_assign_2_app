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
from functools import lru_cache

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
# Suburb data
# ─────────────────────────────────────────────
SUBURB_DATA = [
    # Auckland City / Isthmus
    {"suburb": "Auckland Central",   "lat": -36.8509, "lon": 174.7645, "price": 481000,  "dist_km": 0.5},
    {"suburb": "Parnell",            "lat": -36.8577, "lon": 174.7802, "price": 1850000, "dist_km": 2.0},
    {"suburb": "Ponsonby",           "lat": -36.8544, "lon": 174.7459, "price": 2100000, "dist_km": 3.0},
    {"suburb": "Grey Lynn",          "lat": -36.8650, "lon": 174.7384, "price": 1650000, "dist_km": 4.0},
    {"suburb": "Herne Bay",          "lat": -36.8472, "lon": 174.7333, "price": 3137000, "dist_km": 4.5},
    {"suburb": "Westmere",           "lat": -36.8600, "lon": 174.7200, "price": 2100000, "dist_km": 5.0},
    {"suburb": "Freemans Bay",       "lat": -36.8489, "lon": 174.7518, "price": 1600000, "dist_km": 2.5},
    {"suburb": "Newmarket",          "lat": -36.8710, "lon": 174.7770, "price": 1100000, "dist_km": 3.5},
    {"suburb": "Remuera",            "lat": -36.8783, "lon": 174.7967, "price": 2200000, "dist_km": 5.5},
    {"suburb": "Meadowbank",         "lat": -36.8733, "lon": 174.8183, "price": 1500000, "dist_km": 7.0},
    {"suburb": "St Heliers",         "lat": -36.8683, "lon": 174.8567, "price": 1700000, "dist_km": 10.0},
    {"suburb": "Kohimarama",         "lat": -36.8617, "lon": 174.8417, "price": 1750000, "dist_km": 8.5},
    {"suburb": "Mission Bay",        "lat": -36.8583, "lon": 174.8333, "price": 1650000, "dist_km": 8.0},
    {"suburb": "Mount Eden",         "lat": -36.8780, "lon": 174.7650, "price": 1700000, "dist_km": 6.0},
    {"suburb": "Kingsland",          "lat": -36.8717, "lon": 174.7450, "price": 1400000, "dist_km": 5.0},
    {"suburb": "Sandringham",        "lat": -36.8883, "lon": 174.7433, "price": 1250000, "dist_km": 6.5},
    {"suburb": "Mount Albert",       "lat": -36.8900, "lon": 174.7267, "price": 1200000, "dist_km": 7.5},
    {"suburb": "Waterview",          "lat": -36.8867, "lon": 174.7183, "price": 1050000, "dist_km": 8.0},
    {"suburb": "Point Chevalier",    "lat": -36.8650, "lon": 174.7167, "price": 1500000, "dist_km": 6.0},
    {"suburb": "Epsom",              "lat": -36.8965, "lon": 174.7762, "price": 1850000, "dist_km": 7.0},
    {"suburb": "Royal Oak",          "lat": -36.9050, "lon": 174.7730, "price": 1350000, "dist_km": 8.5},
    {"suburb": "Three Kings",        "lat": -36.9067, "lon": 174.7583, "price": 1300000, "dist_km": 8.0},
    {"suburb": "Onehunga",           "lat": -36.9225, "lon": 174.7839, "price": 1100000, "dist_km": 9.0},
    {"suburb": "Ellerslie",          "lat": -36.9058, "lon": 174.8100, "price": 1300000, "dist_km": 9.5},
    {"suburb": "Glen Innes",         "lat": -36.8819, "lon": 174.8500, "price": 1050000, "dist_km": 12.0},
    {"suburb": "Panmure",            "lat": -36.9017, "lon": 174.8467, "price": 950000,  "dist_km": 11.0},
    {"suburb": "Mount Wellington",   "lat": -36.9100, "lon": 174.8350, "price": 950000,  "dist_km": 11.5},
    {"suburb": "Penrose",            "lat": -36.9233, "lon": 174.8100, "price": 900000,  "dist_km": 10.5},
    {"suburb": "Otahuhu",            "lat": -36.9533, "lon": 174.8383, "price": 800000,  "dist_km": 13.0},
    {"suburb": "Mangere Bridge",     "lat": -36.9600, "lon": 174.7950, "price": 870000,  "dist_km": 13.5},
    {"suburb": "Mangere",            "lat": -36.9817, "lon": 174.7967, "price": 780000,  "dist_km": 15.0},
    # East Auckland
    {"suburb": "Howick",             "lat": -36.9000, "lon": 174.9333, "price": 1150000, "dist_km": 18.0},
    {"suburb": "Pakuranga",          "lat": -36.9000, "lon": 174.9100, "price": 1100000, "dist_km": 16.5},
    {"suburb": "Botany Downs",       "lat": -36.9317, "lon": 174.9183, "price": 1100000, "dist_km": 18.5},
    {"suburb": "Flat Bush",          "lat": -36.9583, "lon": 174.9133, "price": 980000,  "dist_km": 20.0},
    {"suburb": "Beachlands",         "lat": -36.8833, "lon": 175.0000, "price": 1250000, "dist_km": 35.0},
    {"suburb": "Bucklands Beach",    "lat": -36.8767, "lon": 174.9467, "price": 1400000, "dist_km": 20.0},
    {"suburb": "Half Moon Bay",      "lat": -36.8967, "lon": 174.9367, "price": 1350000, "dist_km": 19.0},
    # South Auckland
    {"suburb": "Papatoetoe",         "lat": -36.9833, "lon": 174.8500, "price": 870000,  "dist_km": 19.0},
    {"suburb": "Manukau",            "lat": -36.9933, "lon": 174.8783, "price": 830000,  "dist_km": 21.0},
    {"suburb": "Clover Park",        "lat": -37.0050, "lon": 174.9000, "price": 800000,  "dist_km": 22.0},
    {"suburb": "Takanini",           "lat": -37.0433, "lon": 174.9167, "price": 820000,  "dist_km": 28.0},
    {"suburb": "Papakura",           "lat": -37.0650, "lon": 174.9433, "price": 754500,  "dist_km": 31.0},
    {"suburb": "Pukekohe",           "lat": -37.2000, "lon": 174.9000, "price": 760000,  "dist_km": 50.0},
    {"suburb": "Karaka",             "lat": -37.1167, "lon": 174.9000, "price": 1450000, "dist_km": 38.0},
    # West Auckland
    {"suburb": "Henderson",          "lat": -36.8750, "lon": 174.6317, "price": 900000,  "dist_km": 17.0},
    {"suburb": "Te Atatu South",     "lat": -36.8600, "lon": 174.6600, "price": 1000000, "dist_km": 14.0},
    {"suburb": "Te Atatu Peninsula", "lat": -36.8400, "lon": 174.6517, "price": 1050000, "dist_km": 14.5},
    {"suburb": "Massey",             "lat": -36.8450, "lon": 174.5950, "price": 880000,  "dist_km": 21.0},
    {"suburb": "Waitakere",          "lat": -36.8767, "lon": 174.5650, "price": 885000,  "dist_km": 24.0},
    {"suburb": "Swanson",            "lat": -36.8850, "lon": 174.6133, "price": 870000,  "dist_km": 22.0},
    {"suburb": "Glen Eden",          "lat": -36.9233, "lon": 174.6467, "price": 830000,  "dist_km": 20.0},
    {"suburb": "New Lynn",           "lat": -36.9083, "lon": 174.6867, "price": 950000,  "dist_km": 13.0},
    {"suburb": "Avondale",           "lat": -36.8983, "lon": 174.7133, "price": 1050000, "dist_km": 10.5},
    # North Shore
    {"suburb": "Takapuna",           "lat": -36.7883, "lon": 174.7700, "price": 1900000, "dist_km": 8.0},
    {"suburb": "Devonport",          "lat": -36.8300, "lon": 174.7983, "price": 1950000, "dist_km": 10.0},
    {"suburb": "Northcote",          "lat": -36.8000, "lon": 174.7500, "price": 1300000, "dist_km": 9.0},
    {"suburb": "Birkenhead",         "lat": -36.8133, "lon": 174.7317, "price": 1250000, "dist_km": 9.5},
    {"suburb": "Glenfield",          "lat": -36.7817, "lon": 174.7217, "price": 1100000, "dist_km": 12.0},
    {"suburb": "Browns Bay",         "lat": -36.7200, "lon": 174.7500, "price": 1300000, "dist_km": 19.0},
    {"suburb": "Torbay",             "lat": -36.6933, "lon": 174.7583, "price": 1200000, "dist_km": 22.0},
    {"suburb": "Albany",             "lat": -36.7300, "lon": 174.7000, "price": 1250000, "dist_km": 20.0},
    {"suburb": "Mairangi Bay",       "lat": -36.7333, "lon": 174.7617, "price": 1400000, "dist_km": 18.5},
    {"suburb": "Milford",            "lat": -36.7733, "lon": 174.7633, "price": 1600000, "dist_km": 11.5},
    {"suburb": "Stanley Point",      "lat": -36.8167, "lon": 174.7867, "price": 2100000, "dist_km": 8.0},
    {"suburb": "Birkdale",           "lat": -36.8167, "lon": 174.7133, "price": 1003000, "dist_km": 11.5},
    {"suburb": "Forrest Hill",       "lat": -36.7617, "lon": 174.7383, "price": 1300000, "dist_km": 14.0},
    # Far North / Rodney
    {"suburb": "Orewa",              "lat": -36.5933, "lon": 174.6967, "price": 1200000, "dist_km": 40.0},
    {"suburb": "Silverdale",         "lat": -36.6217, "lon": 174.6700, "price": 1100000, "dist_km": 37.0},
    {"suburb": "Warkworth",          "lat": -36.4000, "lon": 174.6600, "price": 850000,  "dist_km": 65.0},
    {"suburb": "Waiheke Island",     "lat": -36.8000, "lon": 175.1000, "price": 1400000, "dist_km": 35.0},
    {"suburb": "Omaha",              "lat": -36.3417, "lon": 174.7233, "price": 3010000, "dist_km": 75.0},
    {"suburb": "Whangaparaoa",       "lat": -36.6317, "lon": 174.7333, "price": 1050000, "dist_km": 35.0},
    {"suburb": "Coatesville",        "lat": -36.6983, "lon": 174.6683, "price": 2800000, "dist_km": 30.0},
    {"suburb": "Kumeu",              "lat": -36.7817, "lon": 174.5617, "price": 1050000, "dist_km": 30.0},
]


# ─────────────────────────────────────────────
# Load GeoJSON zones (GIS source of truth)
# ─────────────────────────────────────────────
@st.cache_resource
def load_zones():
    parks = gpd.read_file("parks.geojson")
    coast = gpd.read_file("coastline.geojson")

    parks = parks.to_crs(epsg=4326)
    coast = coast.to_crs(epsg=4326)

    parks["zone"] = "reserve"
    coast["zone"] = "sea"

    parks["name"] = parks.get("name", "Park")
    coast["name"] = coast.get("name", "Coast")

    zones = pd.concat([parks, coast], ignore_index=True)

    zones_gdf = gpd.GeoDataFrame(zones, geometry="geometry", crs="EPSG:4326")
    
    # Build spatial indexes for fast queries
    spatial_index = STRtree(zones_gdf.geometry)
    sea_zones_gdf = zones_gdf[zones_gdf["zone"] == "sea"]
    sea_spatial_index = STRtree(sea_zones_gdf.geometry) if len(sea_zones_gdf) > 0 else None
    
    return zones_gdf, spatial_index, sea_zones_gdf, sea_spatial_index

zones_gdf, zones_spatial_index, sea_zones_gdf, sea_spatial_index = load_zones()

# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────
@st.cache_resource
def build_model():
    df = pd.DataFrame(SUBURB_DATA)

    coords = df[["lat", "lon"]].values
    prices = df["price"].values

    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)

    rbf = RBFInterpolator(
        coords_scaled,
        prices,
        kernel="thin_plate_spline",
        smoothing=1e9
    )

    return rbf, scaler, df

rbf, scaler, df = build_model()

# ─────────────────────────────────────────────
# Heatmap
# ─────────────────────────────────────────────
@st.cache_data
def generate_heat(df):
    pmin, pmax = df["price"].min(), df["price"].max()
    rng = np.random.default_rng(42)

    heat = []
    for _, r in df.iterrows():
        intensity = (r["price"] - pmin) / (pmax - pmin)

        for _ in range(25):
            heat.append([
                r["lat"] + rng.normal(0, 0.01),
                r["lon"] + rng.normal(0, 0.01),
                float(intensity)
            ])

    return heat

heat_data = generate_heat(df)

# ─────────────────────────────────────────────
# Spatial logic
# ─────────────────────────────────────────────
@lru_cache(maxsize=512)
def classify_zone(lat, lon):
    """Fast cached zone classification"""
    # Round to 4 decimals (~11m precision) to maximize cache hits
    lat = round(lat, 4)
    lon = round(lon, 4)
    
    pt = Point(lon, lat)
    
    # Use spatial index for fast lookup
    candidates_idx = list(zones_spatial_index.query(pt.envelope))
    
    for idx in candidates_idx:
        if zones_gdf.geometry.iloc[idx].contains(pt):
            row = zones_gdf.iloc[idx]
            return row["zone"], row.get("name", "Zone")
    
    return "residential", "Residential"


@lru_cache(maxsize=512)
def is_ocean(lat, lon):
    """Fast cached ocean check"""
    # Round to 4 decimals (~11m precision) to maximize cache hits
    lat = round(lat, 4)
    lon = round(lon, 4)
    
    if sea_spatial_index is None:
        return False
    
    pt = Point(lon, lat)
    candidates_idx = list(sea_spatial_index.query(pt.envelope))
    
    for idx in candidates_idx:
        if sea_zones_gdf.geometry.iloc[idx].contains(pt):
            return True
    
    return False


def predict_price(lat, lon):
    zone, name = classify_zone(lat, lon)

    if zone != "residential":
        return None, zone, name

    coords = scaler.transform([[lat, lon]])
    price = float(rbf(coords)[0])

    price = np.clip(price, 400000, 5000000)

    return price, zone, name

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.title("Auckland Land Price Estimator")
st.markdown("<p class='subtitle'>Click anywhere on the map</p>", unsafe_allow_html=True)

col_map, col_panel = st.columns([3, 1])

# ─────────────────────────────────────────────
# Map
# ─────────────────────────────────────────────
with col_map:
    m = folium.Map(location=[-36.86, 174.76], zoom_start=11, tiles="CartoDB positron")

    HeatMap(heat_data, radius=18, blur=14).add_to(m)

    for _, r in df.iterrows():
        folium.CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=5,
            fill=True,
            fill_color="#fff",
            color="#000",
            tooltip=f"{r['suburb']} - ${r['price']:,.0f}"
        ).add_to(m)

    # Draw GIS zones (Polygon and MultiPolygon support)
    for z in zones_gdf.itertuples():
        geom = z.geometry

        if geom is None or geom.is_empty:
            continue

        # Handle Polygon
        if geom.geom_type == "Polygon":
            coords = [(y, x) for x, y in geom.exterior.coords]
            folium.Polygon(
                locations=coords,
                color="#4cc9f0",
                fill=True,
                fill_opacity=0.15,
                tooltip=getattr(z, "name", "zone"),
            ).add_to(m)
        
        # Handle MultiPolygon
        elif geom.geom_type == "MultiPolygon":
            for poly in geom.geoms:
                coords = [(y, x) for x, y in poly.exterior.coords]
                folium.Polygon(
                    locations=coords,
                    color="#4cc9f0",
                    fill=True,
                    fill_opacity=0.15,
                    tooltip=getattr(z, "name", "zone"),
                ).add_to(m)
            
    map_data = st_folium(
        m,
        width="100%",
        height=600,
        returned_objects=["last_clicked"]
    )

# ─────────────────────────────────────────────
# Panel
# ─────────────────────────────────────────────
with col_panel:
    st.markdown("### Estimate")

    if map_data and map_data.get("last_clicked"):
        lat = map_data["last_clicked"]["lat"]
        lon = map_data["last_clicked"]["lng"]

        if is_ocean(lat, lon):
            st.warning("Ocean — no pricing model")
        else:
            price, zone, name = predict_price(lat, lon)

            if price is not None:
                st.metric("Estimated Price", f"${price:,.0f}")
            else:
                st.info(f"{name} zone — no residential estimate")

        cbd = np.array([-36.8485, 174.7633])
        dist = np.linalg.norm(np.array([lat, lon]) - cbd) * 111
        st.metric("Distance from CBD", f"{dist:.1f} km")

    else:
        st.info("Click map to estimate land value")

# ─────────────────────────────────────────────
# Data view
# ─────────────────────────────────────────────
with st.expander("Suburb data"):
    st.dataframe(df)