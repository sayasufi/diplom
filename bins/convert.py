import pandas as pd
import pyproj
from math import sin, cos, tan, pi
import matplotlib.pyplot as plt


def convert_geographic_to_projection(lat, lon):
    zone = int(lon / 6.0 + 1)
    a = 6378245.0  # Большая (экваториальная) полуось
    b = 6356863.019  # Малая (полярная) полуось
    e2 = (a ** 2 - b ** 2) / a ** 2  # Эксцентриситет
    n = (a - b) / (a + b)  # Приплюснутость

    F = 1.0  # Масштабный коэффициент
    Lat0 = 0.0  # Начальная параллель (в радианах)
    Lon0 = (zone * 6 - 3) * pi / 180  # Центральный меридиан (в радианах)
    N0 = 0.0  # Условное северное смещение для начальной параллели
    E0 = zone * 1e6 + 500000.0  # Условное восточное смещение для центрального меридиана

    Lat = lat * pi / 180.0
    Lon = lon * pi / 180.0

    # Вычисление переменных для преобразования
    v = a * F * (1 - e2 * (sin(Lat) ** 2)) ** -0.5
    p = a * F * (1 - e2) * (1 - e2 * (sin(Lat) ** 2)) ** -1.5
    n2 = v / p - 1
    M1 = (1 + n + 5.0 / 4.0 * n ** 2 + 5.0 / 4.0 * n ** 3) * (Lat - Lat0)
    M2 = (3 * n + 3 * n ** 2 + 21.0 / 8.0 * n ** 3) * sin(Lat - Lat0) * cos(Lat + Lat0)
    M3 = (15.0 / 8.0 * n ** 2 + 15.0 / 8.0 * n ** 3) * sin(2 * (Lat - Lat0)) * cos(2 * (Lat + Lat0))
    M4 = 35.0 / 24.0 * n ** 3 * sin(3 * (Lat - Lat0)) * cos(3 * (Lat + Lat0))
    M = b * F * (M1 - M2 + M3 - M4)
    I = M + N0
    II = v / 2 * sin(Lat) * cos(Lat)
    III = v / 24 * sin(Lat) * (cos(Lat)) ** 3 * (5 - (tan(Lat) ** 2) + 9 * n2)
    IIIA = v / 720 * sin(Lat) * (cos(Lat) ** 5) * (61 - 58 * (tan(Lat) ** 2) + (tan(Lat) ** 4))
    IV = v * cos(Lat)
    V = v / 6 * (cos(Lat) ** 3) * (v / p - (tan(Lat) ** 2))
    VI = v / 120 * (cos(Lat) ** 5) * (5 - 18 * (tan(Lat) ** 2) + (tan(Lat) ** 4) + 14 * n2 - 58 * (tan(Lat) ** 2) * n2)

    # Вычисление северного и восточного смещения (в метрах)
    N = I + II * (Lon - Lon0) ** 2 + III * (Lon - Lon0) ** 4 + IIIA * (Lon - Lon0) ** 6
    E = E0 + IV * (Lon - Lon0) + V * (Lon - Lon0) ** 3 + VI * (Lon - Lon0) ** 5

    return N, E


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

# df = pd.read_csv("data/ref_xy.csv")
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