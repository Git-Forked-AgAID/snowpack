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
            icol = list(d.columns).index(colname)
            val = d.iloc[i, icol]
            # d.iloc[i, icol] = 123
            if pd.isna(val):
                i2 = i
                while pd.isna(d.iloc[i2, icol]) and i2 < n_rows-1:
                    i2 += 1
                n_nulls = (i2 - i) + 1
                diff = d.iloc[i2, icol] - d.iloc[i-1,icol]
                # input()
                step = diff/n_nulls
                val2 = d.iloc[i-1,icol]
                for j in range(i, i2):
                    val2 += step
                    d.iloc[j, icol] = val2

    for colname in met_cols:
        icol = list(d.columns).index(colname)
        if pd.isna(d.iloc[0,icol]):
            i = 1
            while pd.isna(d.iloc[i,icol]):
                i += 1
            for j in range(i):
                d.iloc[j, icol] = d.iloc[i,icol]

        if pd.isna(d.iloc[n_rows-1][colname]):
            i = n_rows-1
            while pd.isna(d.iloc[i,icol]):
                i -= 1
            for j in range(n_rows-1, i, -1):
                d.iloc[j, icol] = d.iloc[i,icol]

        # print(row)
        # d.rows
    # for i in range(rows):
        # print(rows)
    return d
df_n = gdf#{k:v for k,v in gdf.groupby("Station")}["Slumgullion"]
print(df_n)
print(list(df.columns))
# input()

df_n = df_n.groupby("geometry").apply(runpershard)
print('done with merge')
df_n.to_csv("backfilled2.csv")
print(df_n)
