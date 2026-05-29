import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from scipy.interpolate import RBFInterpolator
from sklearn.preprocessing import StandardScaler
import json

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Auckland Land Price Tool",
    page_icon="🗺️",
    layout="wide",
)

# ── Styling ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
}
h1, h2, h3 {
    font-family: 'DM Serif Display', serif !important;
}
.price-card {
    background: #0f1923;
    border: 1px solid #2a9d8f;
    border-radius: 8px;
    padding: 1.5rem;
    color: #e9f5f3;
    margin-top: 1rem;
}
.price-card h2 {
    color: #2a9d8f;
    font-size: 2rem;
    margin: 0;
}
.price-card .label {
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    color: #7fb3ac;
    text-transform: uppercase;
}
.zone-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 0.5rem;
}
.zone-residential { background: #2a9d8f22; color: #2a9d8f; border: 1px solid #2a9d8f; }
.zone-sea         { background: #22577a22; color: #4cc9f0; border: 1px solid #4cc9f0; }
.zone-reserve     { background: #38761d22; color: #57cc99; border: 1px solid #57cc99; }
.stMetric { background: #0f1923; border-radius: 8px; padding: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ── Data ────────────────────────────────────────────────────────────────────────
# Appendix A data from the assignment, converted to lat/lon.
# Lat/lon estimated from suburb centroids. Swap this out for LINZ data when available.
SUBURB_DATA = [
    {"suburb": "Auckland Central",  "lat": -36.8509, "lon": 174.7645, "price": 480000,  "dist_km": 0.5},
    {"suburb": "Parnell",           "lat": -36.8577, "lon": 174.7802, "price": 1850000, "dist_km": 2.0},
    {"suburb": "Ponsonby",          "lat": -36.8544, "lon": 174.7459, "price": 2100000, "dist_km": 3.0},
    {"suburb": "Grey Lynn",         "lat": -36.8650, "lon": 174.7384, "price": 1900000, "dist_km": 4.0},
    {"suburb": "Herne Bay",         "lat": -36.8472, "lon": 174.7333, "price": 3200000, "dist_km": 4.5},
    {"suburb": "Mount Eden",        "lat": -36.8780, "lon": 174.7650, "price": 1750000, "dist_km": 6.0},
    {"suburb": "Epsom",             "lat": -36.8965, "lon": 174.7762, "price": 1850000, "dist_km": 7.0},
    {"suburb": "Onehunga",          "lat": -36.9225, "lon": 174.7839, "price": 1300000, "dist_km": 9.0},
    {"suburb": "Ellerslie",         "lat": -36.9058, "lon": 174.8100, "price": 1450000, "dist_km": 10.0},
    {"suburb": "Royal Oak",         "lat": -36.9050, "lon": 174.7730, "price": 1400000, "dist_km": 11.0},
    {"suburb": "Glen Innes",        "lat": -36.8819, "lon": 174.8500, "price": 1150000, "dist_km": 12.0},
    {"suburb": "Mount Wellington",  "lat": -36.9100, "lon": 174.8350, "price": 1100000, "dist_km": 13.0},
    {"suburb": "Howick",            "lat": -36.9000, "lon": 174.9333, "price": 1200000, "dist_km": 18.0},
    {"suburb": "Pakuranga",         "lat": -36.9000, "lon": 174.9100, "price": 1150000, "dist_km": 17.0},
    {"suburb": "Henderson",         "lat": -36.8750, "lon": 174.6317, "price": 950000,  "dist_km": 18.0},
    {"suburb": "Te Atatu South",    "lat": -36.8600, "lon": 174.6600, "price": 1050000, "dist_km": 16.0},
    {"suburb": "Massey",            "lat": -36.8450, "lon": 174.5950, "price": 900000,  "dist_km": 22.0},
    {"suburb": "Albany",            "lat": -36.7300, "lon": 174.7000, "price": 1200000, "dist_km": 20.0},
    {"suburb": "Browns Bay",        "lat": -36.7200, "lon": 174.7500, "price": 1300000, "dist_km": 19.0},
    {"suburb": "Northcote",         "lat": -36.8000, "lon": 174.7500, "price": 1250000, "dist_km": 9.0},
    {"suburb": "Papatoetoe",        "lat": -36.9833, "lon": 174.8500, "price": 1000000, "dist_km": 20.0},
    {"suburb": "Manukau",           "lat": -36.9933, "lon": 174.8783, "price": 880000,  "dist_km": 22.0},
    {"suburb": "Papakura",          "lat": -37.0650, "lon": 174.9433, "price": 820000,  "dist_km": 32.0},
    {"suburb": "Takanini",          "lat": -37.0433, "lon": 174.9167, "price": 870000,  "dist_km": 30.0},
    {"suburb": "Pukekohe",          "lat": -37.2000, "lon": 174.9000, "price": 780000,  "dist_km": 50.0},
    {"suburb": "Beachlands",        "lat": -36.8833, "lon": 175.0000, "price": 1200000, "dist_km": 35.0},
    {"suburb": "Orewa",             "lat": -36.5933, "lon": 174.6967, "price": 1250000, "dist_km": 40.0},
    {"suburb": "Warkworth",         "lat": -36.4000, "lon": 174.6600, "price": 850000,  "dist_km": 65.0},
    {"suburb": "Devonport",         "lat": -36.8300, "lon": 174.7983, "price": 1900000, "dist_km": 10.0},
    {"suburb": "Takapuna",          "lat": -36.7883, "lon": 174.7700, "price": 1800000, "dist_km": 8.0},
]

# Non-residential zones (rough polygon centroids for masking)
# Format: (lat_min, lat_max, lon_min, lon_max, zone_type)
NON_RESIDENTIAL_ZONES = [
    # Waitakere Ranges (rough bounding box)
    {"name": "Waitakere Ranges", "lat_min": -37.05, "lat_max": -36.80, "lon_min": 174.45, "lon_max": 174.58, "zone": "reserve"},
    # Hauraki Gulf (sea to the east)
    {"name": "Hauraki Gulf",     "lat_min": -36.90, "lat_max": -36.60, "lon_min": 174.90, "lon_max": 175.20, "zone": "sea"},
    # Manukau Harbour (sea to the west/south)
    {"name": "Manukau Harbour",  "lat_min": -37.10, "lat_max": -36.90, "lon_min": 174.60, "lon_max": 174.75, "zone": "sea"},
    # Waitemata Harbour
    {"name": "Waitemata Harbour","lat_min": -36.84, "lat_max": -36.78, "lon_min": 174.68, "lon_max": 174.82, "zone": "sea"},
    # Cornwall Park / One Tree Hill
    {"name": "Cornwall Park",    "lat_min": -36.905,"lat_max": -36.890,"lon_min": 174.775,"lon_max": 174.790,"zone": "reserve"},
]

# ── Model ───────────────────────────────────────────────────────────────────────
@st.cache_resource
def build_model():
    df = pd.DataFrame(SUBURB_DATA)
    coords = df[["lat", "lon"]].values
    prices = df["price"].values
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)
    rbf = RBFInterpolator(coords_scaled, prices, kernel="thin_plate_spline", smoothing=1e9)
    return rbf, scaler, df

def classify_zone(lat, lon):
    for z in NON_RESIDENTIAL_ZONES:
        if z["lat_min"] <= lat <= z["lat_max"] and z["lon_min"] <= lon <= z["lon_max"]:
            return z["zone"], z["name"]
    return "residential", "Residential"

def predict_price(lat, lon, rbf, scaler):
    zone, zone_name = classify_zone(lat, lon)
    if zone != "residential":
        return None, zone, zone_name
    coords = scaler.transform([[lat, lon]])
    price = float(rbf(coords)[0])
    price = max(400000, min(price, 5000000))  # clamp to sensible range
    return price, zone, zone_name

# ── UI ──────────────────────────────────────────────────────────────────────────
rbf, scaler, df = build_model()

col_title, _ = st.columns([3, 1])
with col_title:
    st.title("Auckland Land Price Estimator")
    st.markdown(
        "<p style='color:#7fb3ac; font-family: DM Mono, monospace; font-size:0.85rem'>"
        "COMP 717 · Q5 · Click anywhere on the map to estimate land value"
        "</p>",
        unsafe_allow_html=True
    )

col_map, col_panel = st.columns([3, 1])

with col_map:
    m = folium.Map(
        location=[-36.86, 174.76],
        zoom_start=11,
        tiles="CartoDB dark_matter",
    )

    # Plot known suburb data points
    for _, row in df.iterrows():
        norm = (row["price"] - df["price"].min()) / (df["price"].max() - df["price"].min())
        r = int(255 * norm)
        g = int(100 + 80 * (1 - norm))
        b = int(150 * (1 - norm))
        color = f"#{r:02x}{g:02x}{b:02x}"
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=7,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            tooltip=f"{row['suburb']}: ${row['price']:,.0f}",
        ).add_to(m)

    # Colour scale legend
    colorscale_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
                background: #0f1923cc; padding: 12px 16px; border-radius: 8px;
                border: 1px solid #2a9d8f; font-family: DM Mono, monospace; font-size: 11px; color: #e9f5f3;">
        <div style="margin-bottom:6px; letter-spacing:0.1em; color:#7fb3ac;">MEDIAN PRICE</div>
        <div style="background: linear-gradient(to right, #006496, #96640a, #ff3200);
                    width: 120px; height: 10px; border-radius: 4px; margin-bottom:4px;"></div>
        <div style="display:flex; justify-content:space-between; width:120px;">
            <span>$480k</span><span>$3.2M</span>
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(colorscale_html))

    map_data = st_folium(m, width="100%", height=560, returned_objects=["last_clicked"])

with col_panel:
    st.markdown("### Estimate")

    if map_data and map_data.get("last_clicked"):
        lat = map_data["last_clicked"]["lat"]
        lon = map_data["last_clicked"]["lng"]

        price, zone, zone_name = predict_price(lat, lon, rbf, scaler)

        zone_class = {
            "residential": "zone-residential",
            "sea": "zone-sea",
            "reserve": "zone-reserve",
        }.get(zone, "zone-residential")

        zone_icon = {"residential": "🏘️", "sea": "🌊", "reserve": "🌿"}.get(zone, "🏘️")

        if price:
            st.markdown(f"""
            <div class="price-card">
                <div class="label">Estimated median price</div>
                <h2>${price:,.0f}</h2>
                <div class="label" style="margin-top:0.75rem">Location</div>
                <div style="font-size:0.8rem; color:#aac; margin-top:0.2rem">
                    {lat:.5f}, {lon:.5f}
                </div>
                <span class="zone-badge {zone_class}">{zone_icon} {zone_name}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="price-card">
                <div class="label">Zone</div>
                <h2 style="font-size:1.4rem">{zone_name}</h2>
                <div style="font-size:0.8rem; color:#aac; margin-top:0.5rem">
                    No residential price estimate available for this zone.
                </div>
                <span class="zone-badge {zone_class}">{zone_icon} {zone_name}</span>
            </div>
            """, unsafe_allow_html=True)

        # Distance from CBD
        cbd = np.array([-36.8485, 174.7633])
        click = np.array([lat, lon])
        dist_deg = np.linalg.norm(click - cbd)
        dist_km = dist_deg * 111
        st.metric("Distance from CBD", f"{dist_km:.1f} km")

    else:
        st.markdown("""
        <div class="price-card" style="opacity:0.5; text-align:center; padding: 2rem 1rem;">
            <div style="font-size:2rem">🗺️</div>
            <div class="label" style="margin-top:0.5rem">Click the map to get an estimate</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Model:** Radial Basis Function (thin-plate spline)")
    st.markdown("**Data:** ~30 Auckland suburb medians, late 2025")
    st.markdown("**Note:** Replace with LINZ data for higher accuracy.")

# ── Data table toggle ────────────────────────────────────────────────────────────
with st.expander("📊 View raw suburb data"):
    st.dataframe(
        df[["suburb", "dist_km", "price"]].rename(columns={
            "suburb": "Suburb", "dist_km": "Dist from CBD (km)", "price": "Median Price ($)"
        }).sort_values("dist_km"),
        use_container_width=True,
        hide_index=True,
    )
