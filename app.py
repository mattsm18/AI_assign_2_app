import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from scipy.interpolate import RBFInterpolator
from sklearn.preprocessing import StandardScaler
from shapely.geometry import Point, Polygon

# ── Page config ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Auckland Land Price Tool", page_icon="🗺️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Mono', monospace; background-color: #f7f5f0; color: #1a1a1a; }
h1, h2, h3 { font-family: 'DM Serif Display', serif !important; color: #1a1a1a; }
.price-card { background: #ffffff; border: 1.5px solid #2a9d8f; border-radius: 8px; padding: 1.5rem;
              color: #1a1a1a; margin-top: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.07); }
.price-card h2 { color: #2a9d8f; font-size: 2rem; margin: 0; font-family: 'DM Serif Display', serif; }
.price-card .label { font-size: 0.7rem; letter-spacing: 0.15em; color: #888; text-transform: uppercase; }
.zone-badge { display: inline-block; padding: 0.25rem 0.75rem; border-radius: 999px; font-size: 0.72rem;
              font-weight: 500; letter-spacing: 0.1em; text-transform: uppercase; margin-top: 0.5rem; }
.zone-residential { background: #e8f5f3; color: #1a7a6e; border: 1px solid #2a9d8f; }
.zone-sea         { background: #e3f4fb; color: #0077a8; border: 1px solid #4cc9f0; }
.zone-reserve     { background: #eaf5ea; color: #2d6a2d; border: 1px solid #57cc99; }
.subtitle { color: #888; font-family: 'DM Mono', monospace; font-size: 0.82rem; margin-top: -0.5rem; }
</style>
""", unsafe_allow_html=True)

# ── Expanded suburb dataset (~80 suburbs, 2025/2026 medians) ────────────────────
# Sources: Opes Partners, OneRoof, MoneyHub, REINZ (late 2025 / early 2026)
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

# ── Zone polygons — accurate coordinates from Google Maps / NZ topo ─────────────
# Each polygon is a list of (lat, lon) tuples tracing the boundary
ZONE_POLYGONS = [
    {
        "name": "Waitematā Harbour", "zone": "sea",
        "coords": [
            (-36.782, 174.660), (-36.782, 174.830), (-36.848, 174.830),
            (-36.848, 174.780), (-36.830, 174.770), (-36.820, 174.750),
            (-36.820, 174.700), (-36.800, 174.680), (-36.782, 174.660),
        ]
    },
    {
        "name": "Manukau Harbour", "zone": "sea",
        "coords": [
            (-36.920, 174.620), (-36.920, 174.780), (-36.960, 174.800),
            (-37.010, 174.790), (-37.050, 174.760), (-37.070, 174.720),
            (-37.060, 174.660), (-37.010, 174.630), (-36.960, 174.620),
            (-36.920, 174.620),
        ]
    },
    {
        "name": "Hauraki Gulf", "zone": "sea",
        "coords": [
            (-36.700, 174.870), (-36.650, 174.920), (-36.620, 175.050),
            (-36.700, 175.150), (-36.820, 175.150), (-36.920, 175.050),
            (-36.950, 174.970), (-36.900, 174.900), (-36.830, 174.860),
            (-36.780, 174.850), (-36.700, 174.870),
        ]
    },
    {
        "name": "Waitakere Ranges", "zone": "reserve",
        "coords": [
            (-36.780, 174.450), (-36.780, 174.570), (-36.840, 174.600),
            (-36.900, 174.590), (-36.960, 174.560), (-37.020, 174.530),
            (-37.050, 174.490), (-37.020, 174.450), (-36.960, 174.430),
            (-36.900, 174.420), (-36.840, 174.430), (-36.780, 174.450),
        ]
    },
    {
        "name": "Hunua Ranges", "zone": "reserve",
        "coords": [
            (-37.080, 175.020), (-37.080, 175.120), (-37.160, 175.130),
            (-37.220, 175.070), (-37.200, 174.980), (-37.140, 174.960),
            (-37.080, 175.020),
        ]
    },
    {
        "name": "Cornwall Park", "zone": "reserve",
        "coords": [
            (-36.893, 174.775), (-36.893, 174.792), (-36.906, 174.792),
            (-36.906, 174.775), (-36.893, 174.775),
        ]
    },
    {
        "name": "Western Springs", "zone": "reserve",
        "coords": [
            (-36.862, 174.718), (-36.862, 174.728), (-36.870, 174.728),
            (-36.870, 174.718), (-36.862, 174.718),
        ]
    },
    {
        "name": "Shakespear Regional Park", "zone": "reserve",
        "coords": [
            (-36.595, 174.780), (-36.570, 174.820), (-36.555, 174.840),
            (-36.575, 174.860), (-36.600, 174.840), (-36.615, 174.810),
            (-36.610, 174.780), (-36.595, 174.780),
        ]
    },
]

# Pre-build Shapely polygons once
ZONE_SHAPES = []
for z in ZONE_POLYGONS:
    # Shapely Polygon takes (lon, lat) i.e. (x, y)
    poly = Polygon([(c[1], c[0]) for c in z["coords"]])
    ZONE_SHAPES.append({"name": z["name"], "zone": z["zone"], "shape": poly})

# ── Model ────────────────────────────────────────────────────────────────────────
@st.cache_resource
def build_model():
    df = pd.DataFrame(SUBURB_DATA)
    coords = df[["lat", "lon"]].values
    prices = df["price"].values
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)
    rbf = RBFInterpolator(coords_scaled, prices, kernel="thin_plate_spline", smoothing=1e9)
    return rbf, scaler, df

@st.cache_data
def generate_heatmap_points(suburb_data):
    df = pd.DataFrame(suburb_data)
    p_min, p_max = df["price"].min(), df["price"].max()
    rng = np.random.default_rng(42)
    heat_data = []
    for _, row in df.iterrows():
        intensity = 0.2 + 0.8 * (row["price"] - p_min) / (p_max - p_min)
        jit_lat = rng.normal(0, 0.012, 30)
        jit_lon = rng.normal(0, 0.012, 30)
        for jl, jo in zip(jit_lat, jit_lon):
            heat_data.append([row["lat"] + jl, row["lon"] + jo, float(intensity)])
    return heat_data

def classify_zone(lat, lon):
    pt = Point(lon, lat)  # Shapely uses (x=lon, y=lat)
    for z in ZONE_SHAPES:
        if z["shape"].contains(pt):
            return z["zone"], z["name"]
    return "residential", "Residential"

def predict_price(lat, lon, rbf, scaler):
    zone, zone_name = classify_zone(lat, lon)
    if zone != "residential":
        return None, zone, zone_name
    coords = scaler.transform([[lat, lon]])
    price = float(rbf(coords)[0])
    price = max(400000, min(price, 5000000))
    return price, zone, zone_name

# ── Build ────────────────────────────────────────────────────────────────────────
rbf, scaler, df = build_model()
heat_data = generate_heatmap_points(SUBURB_DATA)

# ── Header ───────────────────────────────────────────────────────────────────────
col_title, _ = st.columns([3, 1])
with col_title:
    st.title("Auckland Land Price Estimator")
    st.markdown("<p class='subtitle'>COMP 717 · Q5 · Click anywhere on the map to estimate land value</p>",
                unsafe_allow_html=True)

col_map, col_panel = st.columns([3, 1])

with col_map:
    m = folium.Map(location=[-36.86, 174.76], zoom_start=11, tiles="CartoDB positron")

    HeatMap(
        heat_data,
        min_opacity=0.3, max_opacity=0.75,
        radius=18, blur=14,
        gradient={0.0: "#313695", 0.3: "#74add1", 0.5: "#fee090", 0.7: "#f46d43", 1.0: "#a50026"},
    ).add_to(m)

    # Suburb dots
    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]], radius=5,
            color="#1a1a1a", weight=1.5,
            fill=True, fill_color="#ffffff", fill_opacity=0.9,
            tooltip=f"<b>{row['suburb']}</b><br>${row['price']:,.0f}",
        ).add_to(m)

    # Draw zone polygons on map so user can see them
    zone_colours = {"sea": "#4cc9f0", "reserve": "#57cc99"}
    for z in ZONE_POLYGONS:
        folium.Polygon(
            locations=z["coords"],
            color=zone_colours.get(z["zone"], "#aaa"),
            weight=1.5,
            fill=True,
            fill_color=zone_colours.get(z["zone"], "#aaa"),
            fill_opacity=0.15,
            tooltip=z["name"],
        ).add_to(m)

    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;background:rgba(255,255,255,0.92);
                padding:12px 16px;border-radius:8px;border:1px solid #ccc;font-family:monospace;
                font-size:11px;color:#333;box-shadow:0 2px 8px rgba(0,0,0,0.15);">
        <div style="margin-bottom:6px;letter-spacing:0.1em;color:#555;font-weight:600;">MEDIAN PRICE</div>
        <div style="background:linear-gradient(to right,#313695,#74add1,#fee090,#f46d43,#a50026);
                    width:130px;height:10px;border-radius:4px;margin-bottom:5px;"></div>
        <div style="display:flex;justify-content:space-between;width:130px;">
            <span>$480k</span><span>$3.1M</span>
        </div>
        <div style="margin-top:8px;color:#4cc9f0;">▬ Water</div>
        <div style="color:#57cc99;">▬ Reserve/Park</div>
        <div style="color:#777;font-size:10px;margin-top:4px;">● suburb data point</div>
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

    map_data = st_folium(m, width="100%", height=560, returned_objects=["last_clicked"])

with col_panel:
    st.markdown("### Estimate")

    if map_data and map_data.get("last_clicked"):
        lat = map_data["last_clicked"]["lat"]
        lon = map_data["last_clicked"]["lng"]
        price, zone, zone_name = predict_price(lat, lon, rbf, scaler)

        zone_class = {"residential": "zone-residential", "sea": "zone-sea", "reserve": "zone-reserve"}.get(zone, "zone-residential")
        zone_icon  = {"residential": "🏘️", "sea": "🌊", "reserve": "🌿"}.get(zone, "🏘️")

        if price:
            st.markdown(f"""
            <div class="price-card">
                <div class="label">Estimated median price</div>
                <h2>${price:,.0f}</h2>
                <div class="label" style="margin-top:0.75rem">Location</div>
                <div style="font-size:0.8rem;color:#555;margin-top:0.2rem">{lat:.5f}, {lon:.5f}</div>
                <span class="zone-badge {zone_class}">{zone_icon} {zone_name}</span>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="price-card">
                <div class="label">Zone</div>
                <h2 style="font-size:1.4rem">{zone_name}</h2>
                <div style="font-size:0.8rem;color:#555;margin-top:0.5rem">No residential price estimate for this zone.</div>
                <span class="zone-badge {zone_class}">{zone_icon} {zone_name}</span>
            </div>""", unsafe_allow_html=True)

        dist_km = np.linalg.norm(np.array([lat, lon]) - np.array([-36.8485, 174.7633])) * 111
        st.metric("Distance from CBD", f"{dist_km:.1f} km")

    else:
        st.markdown("""
        <div class="price-card" style="opacity:0.5;text-align:center;padding:2rem 1rem;">
            <div style="font-size:2rem">🗺️</div>
            <div class="label" style="margin-top:0.5rem">Click the map to get an estimate</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Model:** RBF — thin-plate spline")
    st.markdown("**Data:** 80 Auckland suburbs, 2025/26")
    st.markdown("**Zones:** Shapely polygon classification")

with st.expander("📊 View suburb data"):
    st.dataframe(
        df[["suburb", "dist_km", "price"]].sort_values("dist_km").rename(columns={
            "suburb": "Suburb", "dist_km": "Dist from CBD (km)", "price": "Median Price ($)"
        }),
        use_container_width=True, hide_index=True,
    )
