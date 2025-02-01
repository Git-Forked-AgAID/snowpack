import pandas as pd
import geopandas as gpd


snotel_df = pd.read_csv('./processed/out.csv')
met_df = pd.read_csv('./processed/met_out.csv')
print('pandas loaded')


met_gdf = gpd.GeoDataFrame(met_df, geometry=gpd.points_from_xy(met_df.lat, met_df.lon))
snotel_gdf = gpd.GeoDataFrame(snotel_df, geometry=gpd.points_from_xy(snotel_df.lat, snotel_df.lon))
print('geopandas loaded')


shards = {k:d for k, d in met_gdf.groupby("date")}
print('shards loaded')
# print(shards['2000-01-01'])

def runpershard(d):
    datekey = d["Date"].values[0]
    print(datekey)
    return gpd.sjoin_nearest(d, shards[datekey], distance_col="distance_from_met")

df_n = snotel_gdf.groupby("Date").apply(runpershard)
print('done with merge')
df_n.to_csv("combined.csv")
print(df_n)
