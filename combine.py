import pandas as pd
import geopandas as gpd

snotel_df = pd.read_csv('./processed/out.csv')
met_df = pd.read_csv('./processed/met_out.csv')
met_df = met_df.loc[met_df["date"]=="1990-01-01"]
met_gdf = gpd.GeoDataFrame(met_df, geometry=gpd.points_from_xy(met_df.lat, met_df.lon))
met_gdf.explore(height=300,width=400)


snotel_gdf = gpd.GeoDataFrame(snotel_df, geometry=gpd.points_from_xy(snotel_df.lat, snotel_df.lon))
print("loaded")
# ---------------
gdf = gpd.sjoin_nearest(snotel_gdf, met_gdf, how="inner", distance_col="dist_to_met")
print(gdf)
