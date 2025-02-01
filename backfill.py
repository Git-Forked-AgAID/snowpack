import pandas as pd
import geopandas as gpd

df = pd.read_csv('./combined.csv')
gdf = gpd.GeoDataFrame(df)

def runpershard(d):
    print(d['Station'].iloc[0])
    # rows = d.rows
    met_cols = ['windspeed', 'tmin', 'tmax', 'srad', 'sph', 'rmin', 'rmax', 'precip']

    n_rows = d.shape[0]
    for i in range(1, n_rows-1):
        for colname in met_cols:
            a = d.iloc[i-1][colname]
            b = d.iloc[i][colname]
            c = d.iloc[i+1][colname]
            try:
                mean = (a + c) / 2
            except:
                print(a, c)
            if pd.isna(b):
                d.iloc[i][colname] = mean

    for colname in met_cols:
        if pd.isna(d.iloc[0][colname]):
            d.iloc[0][colname] = d.iloc[1][colname]
        if pd.isna(d.iloc[n_rows-1][colname]):
            d.iloc[n_rows-1][colname] = d.iloc[n_rows-2][colname]
    return d
        # print(row)
        # d.rows
    # for i in range(rows):
        # print(rows)

df_n = gdf.groupby("geometry").apply(runpershard)
print('done with merge')
df_n.to_csv("backfilled.csv")
print(df_n)
