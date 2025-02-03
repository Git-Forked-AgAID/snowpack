import pandas as pd
import geopandas as gpd

df = pd.read_csv('./processed/backfilled.csv')

clean = pd.DataFrame(df, columns="lat_left,lon_left,date,Station,Elevation,Southness,SWE,windspeed,tmin,tmax,srad,sph,rmin,rmax,precip,distance_from_met".split(','))
clean = clean.rename(columns={
    "lat_left": "lat",
    "lon_left": "lon",
    "SWE": "swe",
    "Station": "station",
    "Elevation": "elevation",
    "Southness": "southness",
    "distance_from_met": "dist_from_met"
})

id_df = pd.read_csv('./data/station_ids.csv')

id_df = pd.DataFrame(id_df, columns=["stationId","stateCode", "countyName", "latitude", "longitude"])
id_df = id_df.rename(columns={
    "stationId": "id",
    "stateCode": "state",
    "countyName": "county",
    "latitude": "lat",
    "longitude": "lon"
})


cols = ['lat', 'lon']
clean = clean.join(id_df.set_index(cols), on=cols)

print(clean)
clean.to_csv("processed/clean_main.csv", index=False)
gdf = gpd.GeoDataFrame(clean, geometry=gpd.points_from_xy(clean.lon, clean.lat))
a = {k:v for k,v in gdf.groupby("date")}["1991-01-01"]
print(a)
a.to_csv("processed/clean_main_day1.csv", index=False)

a = {k:v for k,v in gdf.groupby("station")}
i = 0
for station, gdf in a.items():
    print(station)
    gdf.drop(columns='geometry').to_csv(f"stations/{i}.csv", index=False)
    i += 1

# a.explore("swe")
