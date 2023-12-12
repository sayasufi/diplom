import pandas as pd
import pyproj
from math import sin, cos, tan, pi
import matplotlib.pyplot as plt
import numpy as np

def geographic_to_local(lat, lon):
    x = lon * 20037508.34 / 180
    y = np.log(np.tan((90 + lat) * np.pi / 360)) / (np.pi / 180)
    y = y * 20037508.34 / 180
    return x, y

def local_to_geographic(x, y):
    lon = (x / 20037508.34) * 180
    lat = (y / 20037508.34) * 180
    lat = 180 / np.pi * (2 * np.arctan(np.exp(lat * np.pi / 180)) - np.pi / 2)
    return lat, lon


# gnss = pd.read_csv("data/reference_trajectory.csv")
# df = pd.read_csv("data/ref_xy.csv")
# x = [0] * 10 ** 5
# y = [0] * 10 ** 5
# df["time"] = gnss["time"]
# for i in range(10 ** 5):
#     x[i], y[i] = convert(gnss["lat"][i], gnss["lon"][i])
#     print(i)
# df["x"] = pd.Series(x)
# df["y"] = pd.Series(y)
# df.to_csv("data/ref_xy.csv", index=False)

# df = pd.read_csv("data/rtsln_filter_xy.csv")
# x_coords = df["x"]
# y_coords = df["y"]
# plt.plot(x_coords, y_coords)
# plt.show()
#
# df = pd.read_csv("data/reference_trajectory.csv")
# x_coords = df["lat"]
# y_coords = df["lon"]
# plt.plot(y_coords, x_coords)
# plt.show()