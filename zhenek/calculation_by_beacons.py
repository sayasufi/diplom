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
        (58.050575, 56.322439),
        (57.992717, 56.455636),
        (57.927987, 56.343635),
        (57.985291, 56.207580),
        (57.991502, 56.334067),
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
min_alt = min(alt)

beaconsx = beacons_generate()

for j in range(160):
    n_izm = 1000
    x = [0] * n_izm
    y = [0] * n_izm

    for i in range(n_izm):
        x[i], y[i] = convert_geographic_to_projection(lat[i+n_izm*j], lon[i+n_izm*j])
        logging.info(f"[{j} / 100] Пересчет координат: {i} / {n_izm}")

    x_mayak = [0] * n_izm
    y_mayak = [0] * n_izm
    dist = [0] * n_izm

    distan = pd.read_csv("data/distances.csv")

    for i in range(n_izm):
        x_mayak[i], y_mayak[i], dist[i] = raschet(
            5, x[i], y[i], beaconsx
        )
        distan["dist1"][i+j*n_izm] = math.sqrt(dist[i][0] ** 2 + (alt[i+j*n_izm] - min_alt) ** 2)
        distan["dist2"][i+j*n_izm] = math.sqrt(dist[i][1] ** 2 + (alt[i+j*n_izm] - min_alt) ** 2)
        distan["dist3"][i+j*n_izm] = math.sqrt(dist[i][2] ** 2 + (alt[i+j*n_izm] - min_alt) ** 2)
        distan["dist4"][i+j*n_izm] = math.sqrt(dist[i][3] ** 2 + (alt[i+j*n_izm] - min_alt) ** 2)
        distan["dist5"][i+j*n_izm] = math.sqrt(dist[i][4] ** 2 + (alt[i+j*n_izm] - min_alt) ** 2)

        logging.info(f"[{j} / 100] Расчет координат по маякам: {i} / {n_izm}")


    distan.to_csv("data/distances.csv", index=False)


    rtsln = pd.read_csv("data/rtsln.csv")

    for i in range(n_izm):
        rtsln["lat"][i+j*n_izm], rtsln["lon"][i+j*n_izm] = convert_projection_to_geographic(
            x_mayak[i], y_mayak[i]
        )
        logging.info(f"[{j} / 100] Пересчет координат: {i} / {n_izm}")



    rtsln.to_csv("data/rtsln.csv", index=False)


