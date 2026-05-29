"""
Run this to diagnose why land classification is broken.
Place in the same folder as your geojson files and run with Python.
"""
import os
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union, polygonize
from shapely.validation import make_valid

script_dir = os.path.dirname(os.path.abspath(__file__))

# ── Test point that should definitely be land (Herne Bay) ────────────────────
TEST_LAT, TEST_LON = -36.8472, 174.7333
test_pt = Point(TEST_LON, TEST_LAT)
print(f"Test point (Herne Bay): lon={TEST_LON}, lat={TEST_LAT}")
print()

# ── Inspect coastline.geojson ─────────────────────────────────────────────────
coast_path = os.path.join(script_dir, "coastline.geojson")
if not os.path.exists(coast_path):
    print("ERROR: coastline.geojson not found")
else:
    gdf = gpd.read_file(coast_path).to_crs(epsg=4326)
    print(f"coastline.geojson rows: {len(gdf)}")
    print(f"CRS after reproject: {gdf.crs}")
    print(f"Geometry types: {gdf.geom_type.value_counts().to_dict()}")
    print(f"Bounds: {gdf.total_bounds}")  # minx, miny, maxx, maxy
    print()

    raw = unary_union(gdf.geometry)
    print(f"unary_union type: {raw.geom_type}")
    print(f"unary_union is_valid: {raw.is_valid}")
    print(f"unary_union bounds: {raw.bounds}")
    print()

    if raw.geom_type in ("Polygon", "MultiPolygon"):
        valid = make_valid(raw)
        print(f"Contains Herne Bay (polygon path): {valid.contains(test_pt)}")
    else:
        polys = list(polygonize(raw))
        print(f"polygonize() produced {len(polys)} polygon(s)")
        if polys:
            merged = make_valid(unary_union(polys))
            print(f"Merged polygon bounds: {merged.bounds}")
            print(f"Contains Herne Bay (polygonize path): {merged.contains(test_pt)}")
        else:
            print("polygonize() produced nothing — lines are not closed rings")

    print()
    # Show a sample of the raw coordinates so we can see what the data looks like
    print("First feature geometry summary:")
    first = gdf.geometry.iloc[0]
    print(f"  type: {first.geom_type}")
    if hasattr(first, 'exterior'):
        coords = list(first.exterior.coords)
        print(f"  exterior coord count: {len(coords)}")
        print(f"  first coord (lon, lat): {coords[0]}")
        print(f"  last coord  (lon, lat): {coords[-1]}")
        print(f"  ring closed: {coords[0] == coords[-1]}")
    elif hasattr(first, 'coords'):
        coords = list(first.coords)
        print(f"  coord count: {len(coords)}")
        print(f"  first coord (lon, lat): {coords[0]}")
        print(f"  last coord  (lon, lat): {coords[-1]}")
        print(f"  endpoints match (closed): {coords[0] == coords[-1]}")

print()

# ── Test the suburb-buffer fallback independently ─────────────────────────────
print("── Suburb buffer fallback test ──────────────────────────────────────")
SUBURB_DATA = [
    ("Herne Bay",    -36.8472, 174.7333),
    ("Ponsonby",     -36.8544, 174.7459),
    ("Freemans Bay", -36.8489, 174.7518),
]
from shapely.geometry import MultiPoint
suburb_geoms = [Point(lon, lat).buffer(0.11) for _, lat, lon in SUBURB_DATA]
land_union = make_valid(unary_union(suburb_geoms))
print(f"Suburb buffer union type: {land_union.geom_type}")
print(f"Suburb buffer union bounds: {land_union.bounds}")
print(f"Contains Herne Bay: {land_union.contains(test_pt)}")
print(f"Contains open ocean (-36.5, 175.5): {land_union.contains(Point(175.5, -36.5))}")