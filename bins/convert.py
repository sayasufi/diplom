import pandas as pd
import pyproj


def convert_geographic_to_projection(lat, lon):
    n_zone = lon//6 + 1
    # Определение системы координат СК-42
    crs_src = pyproj.CRS.from_epsg(4740)

    # Определение системы координат WGS84 (географические координаты)
    crs_dst = pyproj.CRS.from_epsg(28400 + n_zone)

    # Создание преобразователя координат
    transformer = pyproj.Transformer.from_crs(crs_src, crs_dst)

    x, y = transformer.transform(lat, lon)
    return x, y


def convert_projection_to_geographic(x, y):
    n_zone = y // 10 ** 6
    # Определение системы координат СК-42
    crs_src = pyproj.CRS.from_epsg(28400 + n_zone)

    # Определение системы координат WGS84 (географические координаты)
    crs_dst = pyproj.CRS.from_epsg(4740)

    # Создание преобразователя координат
    transformer = pyproj.Transformer.from_crs(crs_src, crs_dst, always_xy=True)

    lon, lat = transformer.transform(y, x)
    return lat, lon

gnss = pd.read_csv("data/reference_trajectory.csv")
df = pd.read_csv("data/ref_xy.csv")
x = [0] * 10 ** 5
y = [0] * 10 ** 5
df["time"] = gnss["time"]
for i in range(10 ** 5):
    x[i], y[i] = convert_geographic_to_projection(gnss["lat"][i], gnss["lon"][i])
    print(i)
df["x"] = pd.Series(x)
df["y"] = pd.Series(y)
df.to_csv("data/ref_xy.csv", index=False)

# df = pd.read_csv("data/rtsln_xy.csv")
# x_coords = df["x"]
# y_coords = df["y"]
# plt.plot(y_coords, x_coords)
# plt.show()
#
# df = pd.read_csv("data/reference_trajectory.csv")
# x_coords = df["lat"]
# y_coords = df["lon"]
# plt.plot(y_coords, x_coords)
# plt.show()