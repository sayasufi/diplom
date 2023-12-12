import logging

import pandas as pd

from click import raschet
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
        (55.084559, 38.143594, 192),
        (55.083931, 38.143208, 192),
        (55.082977, 38.147628, 192),
        (55.083469, 38.147993, 192),
        (55.082361, 38.152724, 192),
        (55.081838, 38.152339, 192),
    )

    beacons_xy = []
    for i, j, k in beacons:
        beacons_xy.append((*convert_geographic_to_projection(i, j), k))

    return beacons_xy


gnss = pd.read_csv("data/reference_trajectory.csv")

lat = gnss["lat"]
lon = gnss["lon"]
alt = gnss["alt"]
min_alt = min(alt)

beaconsx = beacons_generate()

for j in range(100):
    n_izm = 1000
    x = [0] * n_izm
    y = [0] * n_izm
    z = [0] * n_izm

    for i in range(n_izm):
        x[i], y[i] = convert_geographic_to_projection(lat[i + n_izm * j], lon[i + n_izm * j])
        z[i] = alt[i + n_izm * j]
        logging.info(f"[{j} / 100] Пересчет координат: {i} / {n_izm}")

    x_mayak = [0] * n_izm
    y_mayak = [0] * n_izm
    dist = [0] * n_izm

    distan = pd.read_csv("data/distances.csv")
    rtsln_xy = pd.read_csv("data/rtsln_xy.csv")

    for i in range(n_izm):
        if i == 0:
            x_mayak[i], y_mayak[i], dist[i] = raschet(
                6, x[i], y[i], z[i], beaconsx, 0, 0, z[i]
            )
        else:
            x_mayak[i], y_mayak[i], dist[i] = raschet(
                6, x[i], y[i], z[i], beaconsx, x_mayak[i - 1], y_mayak[i - 1], z[i]
            )

        rtsln_xy["x"][i + j * n_izm] = x_mayak[i]
        rtsln_xy["y"][i + j * n_izm] = y_mayak[i]
        distan["dist1"][i + j * n_izm] = dist[i][0]
        distan["dist2"][i + j * n_izm] = dist[i][1]
        distan["dist3"][i + j * n_izm] = dist[i][2]
        distan["dist4"][i + j * n_izm] = dist[i][3]
        distan["dist5"][i + j * n_izm] = dist[i][4]
        distan["dist6"][i + j * n_izm] = dist[i][5]
        logging.info(f"[{j} / 100] Расчет координат по маякам: {i} / {n_izm}")

    distan.to_csv("data/distances.csv", index=False)
    rtsln_xy.to_csv("data/rtsln_xy.csv", index=False)

    rtsln = pd.read_csv("data/rtsln.csv")

    for i in range(n_izm):
        rtsln["lat"][i + j * n_izm], rtsln["lon"][i + j * n_izm] = convert_projection_to_geographic(
            x_mayak[i], y_mayak[i]
        )
        logging.info(f"[{j} / 100] Пересчет координат: {i} / {n_izm}")

    rtsln.to_csv("data/rtsln.csv", index=False)
