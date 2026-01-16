import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

# --- INPUTS ---
station_lat, station_lon = 34.0236, -118.2911
buffer_km = 25                                  # choose your radius (e.g., 10–50 km)

perims = gpd.read_file("California_Historic_Fire_Perimeters.geojson")
climate = pd.read_csv("LA_climate_with_PRCP_rolling.csv", parse_dates=["date"])

# Project to a metric CRS for accurate buffering (California Albers)
metric_crs = "EPSG:3310"
perims_m = perims.to_crs(metric_crs)

# Make a station buffer
station_gdf = gpd.GeoDataFrame(geometry=[Point(station_lon, station_lat)], crs="EPSG:4326").to_crs(metric_crs)
station_buf = station_gdf.buffer(buffer_km * 1000).iloc[0]  # meters

# Intersect perimeters with buffer
perims_near = perims_m[perims_m.geometry.intersects(station_buf)].copy()

# Dates
for col in ["ALARM_DATE", "CONT_DATE", "DISCOVERY_DATE", "CONTAINMENT_DATE"]:
    if col in perims_near.columns:
        perims_near[col] = pd.to_datetime(perims_near[col], errors="coerce")

start_col = "ALARM_DATE" if "ALARM_DATE" in perims_near.columns else "DISCOVERY_DATE"
end_col   = "CONT_DATE" if "CONT_DATE" in perims_near.columns else ("CONTAINMENT_DATE" if "CONTAINMENT_DATE" in perims_near.columns else None)

perims_near = perims_near.dropna(subset=[start_col])
if end_col is None:
    perims_near[end_col] = perims_near[start_col]

# Explode to daily flags
rows = []
for _, r in perims_near.iterrows():
    s = r[start_col].normalize()
    e = r[end_col].normalize() if pd.notnull(r[end_col]) else r[start_col].normalize()

    if pd.isna(e) or e < s:
        e = s
    rows.extend(pd.date_range(s, e, freq="D"))

fire_days = (
    pd.DataFrame({"date": pd.Series(rows, dtype="datetime64[ns]")})
      .drop_duplicates()
      .assign(fire_occurred=1)
)

out = (climate.merge(fire_days, on="date", how="left")
              .assign(fire_occurred=lambda d: d["fire_occurred"].fillna(0).astype(int)))
out.to_csv("LA_climate_fire_labels_stationBuffer.csv", index=False)
print("Saved: LA_climate_fire_labels_stationBuffer.csv")
