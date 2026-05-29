"""
Run this once to generate auckland_prices.html.
Requires: parks.geojson and coastline.geojson in the same folder.
pip install folium scipy scikit-learn geopandas shapely numpy
"""

import json, os
import numpy as np
import folium
from folium.plugins import HeatMap
from scipy.interpolate import RBFInterpolator
from sklearn.preprocessing import StandardScaler
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.validation import make_valid

# ── Suburb price data ($/m²) ──────────────────────────────────────────────────

SUBURB_DATA = [
    ("Auckland Central",   -36.8509, 174.7645,  3_200), 
    ("Parnell",            -36.8577, 174.7802,  3_800),
    ("Ponsonby",           -36.8544, 174.7459,  4_200),
    ("Grey Lynn",          -36.8650, 174.7384,  3_600),
    ("Herne Bay",          -36.8472, 174.7333,  5_100),
    ("Westmere",           -36.8600, 174.7200,  3_500),
    ("Freemans Bay",       -36.8489, 174.7518,  4_000),
    ("Newmarket",          -36.8710, 174.7770,  3_400),
    ("Remuera",            -36.8783, 174.7967,  3_700),
    ("Meadowbank",         -36.8733, 174.8183,  2_100),
    ("St Heliers",         -36.8683, 174.8567,  2_800),
    ("Kohimarama",         -36.8617, 174.8417,  2_900),
    ("Mission Bay",        -36.8583, 174.8333,  3_100),
    ("Mount Eden",         -36.8780, 174.7650,  3_300),
    ("Kingsland",          -36.8717, 174.7450,  2_800),
    ("Sandringham",        -36.8883, 174.7433,  2_300),
    ("Mount Albert",       -36.8900, 174.7267,  2_200),
    ("Waterview",          -36.8867, 174.7183,  1_800),
    ("Point Chevalier",    -36.8650, 174.7167,  2_600),
    ("Epsom",              -36.8965, 174.7762,  3_200), 
    ("Royal Oak",          -36.9050, 174.7730,  2_000),
    ("Three Kings",        -36.9067, 174.7583,  1_950),
    ("Onehunga",           -36.9225, 174.7839,  1_800),
    ("Ellerslie",          -36.9058, 174.8100,  2_100),
    ("Glen Innes",         -36.8819, 174.8500,  1_650), 
    ("Panmure",            -36.9017, 174.8467,  1_500),
    ("Mount Wellington",   -36.9100, 174.8350,  1_550),
    ("Penrose",            -36.9233, 174.8100,  1_400),
    ("Otahuhu",            -36.9533, 174.8383,  1_100),
    ("Mangere Bridge",     -36.9600, 174.7950,  1_450),
    ("Mangere",            -36.9817, 174.7967,  1_050),
    ("Howick",             -36.9000, 174.9333,  1_600),
    ("Pakuranga",          -36.9000, 174.9100,  1_500),
    ("Botany Downs",       -36.9317, 174.9183,  1_450),
    ("Flat Bush",          -36.9583, 174.9133,  1_300),
    ("Beachlands",         -36.8833, 175.0000,  1_100),
    ("Bucklands Beach",    -36.8767, 174.9467,  1_950),
    ("Half Moon Bay",      -36.8967, 174.9367,  1_800),
    ("Papatoetoe",         -36.9833, 174.8500,  1_150),
    ("Manukau",            -36.9933, 174.8783,  1_200),
    ("Clover Park",        -37.0050, 174.9000,  1_050),
    ("Takanini",           -37.0433, 174.9167,    950),
    ("Papakura",           -37.0650, 174.9433,    850),
    ("Pukekohe",           -37.2000, 174.9000,    650),
    ("Karaka",             -37.1167, 174.9000,    450),
    ("Henderson",          -36.8750, 174.6317,  1_250),
    ("Te Atatu South",     -36.8600, 174.6600,  1_400),
    ("Te Atatu Peninsula", -36.8400, 174.6517,  1_650),
    ("Massey",             -36.8450, 174.5950,  1_150),
    ("Waitakere",          -36.8767, 174.5650,    450), 
    ("Swanson",            -36.8850, 174.6133,    950),
    ("Glen Eden",          -36.9233, 174.6467,  1_150),
    ("New Lynn",           -36.9083, 174.6867,  1_450),
    ("Avondale",           -36.8983, 174.7133,  1_600),
    ("Takapuna",           -36.7883, 174.7700,  3_600),
    ("Devonport",          -36.8300, 174.7983,  3_400),
    ("Northcote",          -36.8000, 174.7500,  2_000),
    ("Birkenhead",         -36.8133, 174.7317,  1_950),
    ("Glenfield",          -36.7817, 174.7217,  1_500),
    ("Browns Bay",         -36.7200, 174.7500,  1_800),
    ("Torbay",             -36.6933, 174.7583,  1_600),
    ("Albany",             -36.7300, 174.7000,  1_650),
    ("Mairangi Bay",       -36.7333, 174.7617,  2_100),
    ("Milford",            -36.7733, 174.7633,  2_500),
    ("Stanley Point",      -36.8167, 174.7867,  3_500),
    ("Birkdale",           -36.8167, 174.7133,  1_350),
    ("Forrest Hill",       -36.7617, 174.7383,  1_850),
    ("Orewa",              -36.5933, 174.6967,  1_650),
    ("Silverdale",         -36.6217, 174.6700,  1_350),
    ("Warkworth",          -36.4000, 174.6600,    650),
    ("Waiheke Island",     -36.8000, 175.1000,  1_200),
    ("Omaha",              -36.3417, 174.7233,  3_100),
    ("Whangaparaoa",       -36.6317, 174.7333,  1_250),
    ("Coatesville",        -36.6983, 174.6683,    220), 
    ("Kumeu",              -36.7817, 174.5617,    550),
]

CBD = [-36.8485, 174.7633]

# ── Train RBF model ──────────────────────────────────────────────────────────

coords = np.array([[r[1], r[2]] for r in SUBURB_DATA])
prices = np.array([r[3] for r in SUBURB_DATA], dtype=float)

scaler = StandardScaler()
coords_scaled = scaler.fit_transform(coords)

rbf = RBFInterpolator(coords_scaled, prices, kernel="thin_plate_spline", smoothing=0.0)

# Build a fine prediction grid
lat_range = np.linspace(-37.25, -36.35, 300)
lon_range = np.linspace(174.50, 175.10, 300)
grid_lats, grid_lons = np.meshgrid(lat_range, lon_range)
grid_pts = np.column_stack([grid_lats.ravel(), grid_lons.ravel()])
grid_pts_scaled = scaler.transform(grid_pts)
grid_prices = rbf(grid_pts_scaled).reshape(grid_lats.shape)

# FIX: Clamp adjusted to realistic land value per sqm range ($100 - $6,000)
grid_prices = np.clip(grid_prices, 100, 6_000)

# ── Heatmap data (for folium HeatMap layer) ──────────────────────────────────

log_p = np.log(prices)
norm  = (log_p - log_p.min()) / (log_p.max() - log_p.min())
rng   = np.random.default_rng(42)
n     = len(SUBURB_DATA) * 40
hl    = np.repeat(coords[:, 0], 40) + rng.normal(0, 0.009, n)
hlon  = np.repeat(coords[:, 1], 40) + rng.normal(0, 0.009, n)
hi    = np.repeat(norm, 40)
heat  = np.column_stack([hl, hlon, hi]).tolist()

# ── Load GeoJSON ─────────────────────────────────────────────────────────────

script_dir = os.path.dirname(os.path.abspath(__file__))

def load_geojson(fname):
    path = os.path.join(script_dir, fname)
    if not os.path.exists(path):
        print(f"WARNING: {fname} not found — skipping layer.")
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def load_gdf(fname):
    path = os.path.join(script_dir, fname)
    if not os.path.exists(path):
        return None
    return gpd.read_file(path).to_crs(epsg=4326)

parks_geojson     = load_geojson("parks.geojson")
coastline_geojson = load_geojson("coastline.geojson")
parks_gdf         = load_gdf("parks.geojson")

# ── Build land mask ───────────────────────────────────────────────────────────

print("Building land mask from suburb buffers...")
suburb_geoms = [Point(lon, lat).buffer(0.11) for _, lat, lon, _ in SUBURB_DATA]
land_union = make_valid(unary_union(suburb_geoms))

park_union = unary_union(parks_gdf.geometry) if parks_gdf is not None else None

# ── Classify grid cells ───────────────────────────────────────────────────────

print("Classifying grid cells...")
classify_flat = np.zeros(len(grid_pts), dtype=np.int8)

for i, (lat, lon) in enumerate(grid_pts):
    if land_union.contains(Point(lon, lat)):
        classify_flat[i] = 1

if park_union is not None:
    for i, (lat, lon) in enumerate(grid_pts):
        if park_union.contains(Point(lon, lat)):
            classify_flat[i] = 2

classify_grid = classify_flat.reshape(grid_lats.shape)
print("Done.")

# ── Serialise lookup data for JS ─────────────────────────────────────────────

# FIX: Preserve actual dollar values directly as integers instead of dividing by 1000
prices_grid_rounded = np.round(grid_prices).astype(int).tolist()
classify_list = classify_grid.astype(int).tolist()
lat_list = [round(float(x), 4) for x in lat_range]
lon_list = [round(float(x), 4) for x in lon_range]

suburb_js = [
    {"name": r[0], "lat": r[1], "lon": r[2], "price": r[3]}
    for r in SUBURB_DATA
]

cbd_lat, cbd_lon = CBD

NAME_CANDIDATES = ["name", "Name", "NAME", "SITE", "SITEDESCRIPTION", "DESCRIPTION", "title"]
parks_name_field = None
if parks_geojson:
    sample_props = {}
    for f in parks_geojson.get("features", []):
        sample_props = f.get("properties") or {}
        if sample_props:
            break
    for candidate in NAME_CANDIDATES:
        if candidate in sample_props:
            parks_name_field = candidate
            break

# ── Build Folium map ─────────────────────────────────────────────────────────

m = folium.Map(location=[-36.86, 174.76], zoom_start=11, tiles="CartoDB positron")

HeatMap(heat, radius=22, blur=18, min_opacity=0.3).add_to(m)

if parks_geojson:
    folium.GeoJson(
        parks_geojson,
        name="Parks",
        style_function=lambda _: {
            "fillColor": "#22c55e", "color": "#16a34a",
            "weight": 1, "fillOpacity": 0.35,
        },
        tooltip=folium.GeoJsonTooltip(fields=[parks_name_field]) if parks_name_field else None,
    ).add_to(m)

if coastline_geojson:
    folium.GeoJson(
        coastline_geojson,
        name="Coastline",
        style_function=lambda _: {
            "fillColor": "none", "color": "#1d4ed8",
            "weight": 1.5, "fillOpacity": 0,
        },
    ).add_to(m)

for name, lat, lon, price in SUBURB_DATA:
    folium.CircleMarker(
        location=[lat, lon], radius=4,
        fill=True, fill_color="#fff", fill_opacity=0.9,
        color="#333", weight=1,
        tooltip=f"{name} — ${price:,.0f} / m²",
    ).add_to(m)

# ── Inject JS for click-to-estimate ─────────────────────────────────────────

JS = f"""
<script>
const LAT_LIST  = {json.dumps(lat_list)};
const LON_LIST  = {json.dumps(lon_list)};
const PRICES    = {json.dumps(prices_grid_rounded)};  // Actual $/m² per cell [lon_idx][lat_idx]
const CLASSIFY  = {json.dumps(classify_list)};         // 0=sea 1=residential 2=park
const SUBURBS   = {json.dumps(suburb_js)};
const CBD_LAT   = {cbd_lat};
const CBD_LON   = {cbd_lon};

function nearestIdx(arr, val) {{
    let best = 0, bestD = Math.abs(arr[0] - val);
    for (let i = 1; i < arr.length; i++) {{
        const d = Math.abs(arr[i] - val);
        if (d < bestD) {{ bestD = d; best = i; }}
    }}
    return best;
}}

function haversineKm(lat1, lon1, lat2, lon2) {{
    const R = 6371, dLat = (lat2-lat1)*Math.PI/180, dLon = (lon2-lon1)*Math.PI/180;
    const a = Math.sin(dLat/2)**2 + Math.cos(lat1*Math.PI/180)*Math.cos(lat2*Math.PI/180)*Math.sin(dLon/2)**2;
    return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
}}

function nearestSuburb(lat, lon) {{
    let best = SUBURBS[0], bestD = 1e9;
    for (const s of SUBURBS) {{
        const d = haversineKm(lat, lon, s.lat, s.lon);
        if (d < bestD) {{ bestD = d; best = s; }}
    }}
    return {{name: best.name, km: bestD.toFixed(1)}};
}}

const panel = document.createElement('div');
panel.id = 'price-panel';
panel.innerHTML = '<div class="ph">Auckland Land Price</div><div id="ph-body"><p class="hint">Click anywhere on the map</p></div>';
panel.style.cssText = `
    position:fixed; top:16px; right:16px; z-index:9999;
    background:#1a1a1a; color:#f0ede8; border-radius:12px;
    padding:1.25rem 1.5rem; width:240px; font-family:monospace;
    box-shadow:0 4px 24px rgba(0,0,0,0.4); font-size:13px;
`;
document.body.appendChild(panel);

const style = document.createElement('style');
style.textContent = `
    .ph {{ font-size:15px; font-weight:bold; color:#f0ede8; border-bottom:1px solid #333; padding-bottom:8px; margin-bottom:12px; }}
    .hint {{ color:#555; font-size:12px; text-align:center; padding:8px 0; }}
    .badge {{ display:inline-block; padding:2px 10px; border-radius:4px; font-size:11px; font-weight:bold; text-transform:uppercase; margin-bottom:10px; }}
    .b-res {{ background:#c8f560; color:#111; }}
    .b-park {{ background:#5eead4; color:#111; }}
    .b-sea {{ background:#60a5fa; color:#111; }}
    .price {{ font-size:1.6rem; font-weight:bold; color:#c8f560; line-height:1.1; }}
    .price-range {{ font-size:11px; color:#888; margin-bottom:10px; margin-top:2px; }}
    .lbl {{ font-size:10px; text-transform:uppercase; letter-spacing:0.1em; color:#555; margin-top:8px; }}
    .val {{ font-size:13px; color:#f0ede8; }}
    .sea-msg {{ color:#60a5fa; font-size:14px; }}
    .park-msg {{ color:#5eead4; font-size:14px; }}
`;
document.head.appendChild(style);

window.addEventListener('load', function() {{
    setTimeout(function() {{
        let leafletMap = null;
        for (const k of Object.keys(window)) {{
            try {{
                if (window[k] && typeof window[k].on === 'function' && window[k].getCenter) {{
                    leafletMap = window[k]; break;
                }}
            }} catch(e) {{}}
        }}
        if (!leafletMap) {{ console.warn('Could not find Leaflet map'); return; }}

        leafletMap.on('click', function(e) {{
            const lat = e.latlng.lat, lon = e.latlng.lng;
            const li  = nearestIdx(LAT_LIST, lat);
            const loi = nearestIdx(LON_LIST, lon);
            const zone  = CLASSIFY[loi][li];
            const price = PRICES[loi][li]; // FIX: Directly maps to true $/m² raw integer
            const cbd_km = haversineKm(lat, lon, CBD_LAT, CBD_LON).toFixed(1);
            const sub    = nearestSuburb(lat, lon);

            let html = '';
            if (zone === 0) {{
                html = `<span class="badge b-sea">Ocean</span><br>
                        <span class="sea-msg">No pricing available</span>`;
            }} else if (zone === 2) {{
                html = `<span class="badge b-park">Park / Reserve</span><br>
                        <span class="park-msg">No residential pricing</span>`;
            }} else {{
                // FIX: Removed internal multiplication by 1000, added / m² metric tags
                const lo    = Math.round(price * 0.88);
                const hi    = Math.round(price * 1.12);
                const fmt   = v => '$' + v.toLocaleString();
                html = `<span class="badge b-res">Residential</span><br>
                        <div class="price">${{fmt(price)}} / m²</div>
                        <div class="price-range">${{fmt(lo)}} – ${{fmt(hi)}} / m² range</div>`;
            }}

            html += `
                <div class="lbl">Distance to CBD</div>
                <div class="val">${{cbd_km}} km</div>
                <div class="lbl">Nearest suburb</div>
                <div class="val">${{sub.name}} <span style="color:#666;font-size:11px">${{sub.km}} km</span></div>
                <div class="lbl">Coordinates</div>
                <div class="val" style="font-size:11px">${{lat.toFixed(4)}}, ${{lon.toFixed(4)}}</div>
            `;
            document.getElementById('ph-body').innerHTML = html;
        }});
    }}, 800);
}});
</script>
"""

m.get_root().html.add_child(folium.Element(JS))

out = os.path.join(script_dir, "auckland_prices.html")
m.save(out)
print(f"\nSaved → {out}")
print("Open that file in any browser. No server needed.")