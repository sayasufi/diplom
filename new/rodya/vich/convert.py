import numpy as np
import pandas as pd


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


# print(geographic_to_local(55.08702067774778, 38.1481475549193))
#
# df = pd.read_csv("ref.csv")
# rtsln = pd.DataFrame(columns=['time', 'x', 'y'], index=range(10000))
# x = [0] * 10 ** 4
# y = [0] * 10 ** 4
# rtsln["time"] = df["time"]
# for i in range(10 ** 4):
#     x[i], y[i] = geographic_to_local(df["lat"][i], df["lon"][i])
#     print(i)
# rtsln["x"] = pd.Series(x)
# rtsln["y"] = pd.Series(y)
# rtsln.to_csv("data/rodya/vzlet_ref_xy.csv", index=False)

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
