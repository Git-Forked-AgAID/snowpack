import pandas as pd
import geopandas as gpd

station_df = pd.read_csv('./data/swe_data/Station_Info.csv')
swe_df = pd.read_csv("./data/swe_data/SWE_values_all.csv")

cols = ['Latitude', 'Longitude']
snotel_df = swe_df.join(station_df.set_index(cols), on=cols)
print(snotel_df)
snotel_df = snotel_df.rename(columns={"Latitude":"lat", "Longitude":"lon"})
snotel_df.to_csv("out.csv")
snotel_gdf = gpd.GeoDataFrame(snotel_df, geometry=gpd.points_from_xy(snotel_df.lat, snotel_df.lon))

swe_df = pd.read_csv("./data/swe_data/SWE_values_all.csv")


# ---------------

met_df = pd.read_csv("./data/meteorological_data/Modified_Output_Rmax.csv")
met_df = met_df.rename(columns={'variable_value':'rmax'})
for colname in ["Rmin", "SRAD", "precip", "tmax", "tmin", "windspeed", "SPH"]:
    print(colname)
    cols = ['lat', 'lon', 'date']
    met_df2 = pd.read_csv("./data/meteorological_data/Modified_Output_"+colname+".csv")
    met_df2 = met_df2.rename(columns={"variable_value":colname.lower()})
    met_df = met_df.join(met_df2.set_index(cols), on=cols)

print(met_df)
met_gdf = gpd.GeoDataFrame(met_df, geometry=gpd.points_from_xy(met_df.lat, met_df.lon))
met_gdf.plot()
met_df.to_csv("met_out.csv")
input()
