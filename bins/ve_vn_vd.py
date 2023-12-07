import pandas as pd

from convert import (
    convert_geographic_to_projection,
    convert_projection_to_geographic,
)

df = pd.read_csv("data/rtsln.csv")
df_xy = pd.DataFrame()
df_xy["time"] = df["time"]
lat = df["lat"]
lon = df["lon"]
df_xy["x"] = df["lat"]
df_xy["y"] = df["lon"]
for i in range(lat.shape[0]):
    df_xy["x"][i], df_xy["y"][i] = convert_geographic_to_projection(df["lat"][i], df["lon"][i])
    print(i)
df_xy["alt"] = df["alt"]

df_xy.to_csv("data/rtsln_xy.csv", index=False)