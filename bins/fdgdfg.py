import logging
import math

import folium

from click import raschet
import pandas as pd
from convert import (
    convert_geographic_to_projection,
    convert_projection_to_geographic,
)

logger = logging.getLogger()
logger.setLevel(logging.NOTSET)

formatter = logging.StreamHandler()
formatter.setLevel(logging.INFO)
formatter.setFormatter(logging.Formatter("%(asctime)s\t%(message)s"))
logger.addHandler(formatter)


def beacons_generate():
    beacons = (
        (55.084559, 38.143594),
        (55.083931, 38.143208),
        (55.082977, 38.147628),
        (55.083469, 38.147993),
        (55.082361, 38.152724),
        (55.081838, 38.152339),
    )

    beacons_xy = []
    for i, j in beacons:
        beacons_xy.append(convert_geographic_to_projection(i, j))

    # lat = []
    # lon = []
    # for i, j in beacons:
    #     lat.append(i)
    #     lon.append(j)
    # places = [[x[0], x[1]] for x in zip(lat, lon)]
    # m = folium.Map(places[0], tiles="OpenStreetMap", zoom_start=13)
    # for place in places:
    #     folium.Marker(place).add_to(m)
    # m.save("map1.html")

    return beacons_xy


gnss = pd.read_csv("data/reference_trajectory.csv")

lat = gnss["lat"]
lon = gnss["lon"]
alt = gnss["alt"]

# n_izm = lat.shape[0]
n_izm = 100
x = [0] * n_izm
y = [0] * n_izm

for i in range(n_izm):
    x[i], y[i] = convert_geographic_to_projection(lat[i], lon[i])
    logging.info(f"Пересчет координат: {i} / {n_izm}")

x_mayak = [0] * n_izm
y_mayak = [0] * n_izm
dist = [0] * n_izm

dist1 = [0] * n_izm
dist2 = [0] * n_izm
dist3 = [0] * n_izm
dist4 = [0] * n_izm
dist5 = [0] * n_izm
dist6 = [0] * n_izm

for i in range(n_izm):
    x_mayak[i], y_mayak[i], dist[i] = raschet(
        6, x[i], y[i], beacons_generate()
    )
    dist1[i] = math.sqrt(dist[i][0] ** 2 + alt[i] ** 2)
    dist2[i] = math.sqrt(dist[i][1] ** 2 + (alt[i] - alt[0]) ** 2)
    dist3[i] = math.sqrt(dist[i][2] ** 2 + (alt[i] - alt[0]) ** 2)
    dist4[i] = math.sqrt(dist[i][3] ** 2 + (alt[i] - alt[0]) ** 2)
    dist5[i] = math.sqrt(dist[i][4] ** 2 + (alt[i] - alt[0]) ** 2)
    dist6[i] = math.sqrt(dist[i][5] ** 2 + (alt[i] - alt[0]) ** 2)
    logging.info(f"Расчет координат по маякам: {i} / {n_izm}")

distan = pd.DataFrame()
distan["time"] = gnss["time"]
distan["dist1"] = pd.Series(dist1)
distan["dist2"] = pd.Series(dist2)
distan["dist3"] = pd.Series(dist3)
distan["dist4"] = pd.Series(dist4)
distan["dist5"] = pd.Series(dist5)
distan["dist6"] = pd.Series(dist6)
distan.to_csv("data/distances.csv", index=False)


lat_mayak = [0] * n_izm
lon_mayak = [0] * n_izm

for i in range(n_izm):
    lat_mayak[i], lon_mayak[i] = convert_projection_to_geographic(
        x_mayak[i], y_mayak[i]
    )
    logging.info(f"Пересчет координат: {i} / {n_izm}")

rtsln = pd.DataFrame()
rtsln["time"] = gnss["time"]
rtsln["lat"] = pd.Series(lat_mayak)
rtsln["lon"] = pd.Series(lon_mayak)
rtsln.to_csv("data/rtsln.csv", index=False)


# places = [[x[0], x[1]] for x in zip(lat_mayak, lon_mayak)]
# m = folium.Map(places[0], tiles='OpenStreetMap', zoom_start=13)
# polyline = folium.PolyLine(locations=places, color='blue', weight=3)
# polyline.add_to(m)
# m.save('map_mayak.html')
