import pandas as pd
import geopandas as gpd

snotel_df = pd.read_csv('./data/test/Test_InputData_staticVars_2017_2019.csv')
snotel_df.columns = map(str.lower, snotel_df.columns)
met_df = pd.read_csv('./data/test/Test_InputData_dynamicVars_2017_2019.csv')
met_df.columns = map(str.lower, met_df.columns)
print(met_df)
print(snotel_df)


cols = ['lat', 'lon']
out = snotel_df.join(met_df.set_index(cols), on=cols)
print(out)

s = pd.read_csv('./data/swe_data/Station_Info.csv')
s["lat"] = s["Latitude"]
s["lon"] = s['Longitude']
print(s)
s = gpd.GeoDataFrame(s, geometry=gpd.points_from_xy(s.lat, s.lon))
s.cx[]


# met_df = met_df.loc[met_df["date"]=="1990-01-01"]
# met_gdf = gpd.GeoDataFrame(met_df, geometry=gpd.points_from_xy(met_df.lat, met_df.lon))
# exp = met_gdf.explore(height=300,width=400)
# outfp = r"processed/explore.html"
# exp.save(outfp)

# snotel_gdf = gpd.GeoDataFrame(snotel_df, geometry=gpd.points_from_xy(snotel_df.lat, snotel_df.lon))
# print("loaded")
# ---------------
# gdf = gpd.sjoin_nearest(snotel_gdf, met_gdf, how="inner", distance_col="dist_to_met")
# print(gdf)
