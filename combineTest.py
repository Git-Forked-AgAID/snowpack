import pandas as pd
import geopandas as gpd

snotel_df = pd.read_csv('./data/test/Test_InputData_staticVars_2017_2019.csv')
snotel_df.columns = map(str.lower, snotel_df.columns)
met_df = pd.read_csv('./data/test/Test_InputData_dynamicVars_2017_2019.csv')
met_df.columns = map(str.lower, met_df.columns)
print(met_df)
print(snotel_df)

o = geopandas.sjoin(snotel_df, met_df)
print(o)
# cols = ['lat', 'lon']
# out = snotel_df.join(met_df.set_index(cols), on=cols)
