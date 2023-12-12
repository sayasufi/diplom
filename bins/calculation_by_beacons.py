import logging

import pandas as pd

from click import raschet
from convert import *

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
        beacons_xy.append((*geographic_to_local(i, j), k))

    return beacons_xy


gnss = pd.read_csv("data/reference_trajectory.csv")

lat = gnss["lat"]
lon = gnss["lon"]
alt = gnss["alt"]
min_alt = min(alt)

beaconsx = beacons_generate()

n_izm = lat.shape[0]

x = [0] * n_izm
y = [0] * n_izm
z = [0] * n_izm

x_mayak = [0] * n_izm
y_mayak = [0] * n_izm
dist = [0] * n_izm

distan = pd.read_csv("data/distances.csv")
rtsln_xy = pd.read_csv("data/rtsln_xy.csv")
rtsln = pd.read_csv("data/rtsln.csv")

for i in range(n_izm):
    x[i], y[i] = geographic_to_local(lat[i], lon[i])
    z[i] = alt[i]

    if i == 0:
        x_mayak[i], y_mayak[i], dist[i] = raschet(
            6, x[i], y[i], z[i], beaconsx, 0, 0, z[i]
        )
    else:
        x_mayak[i], y_mayak[i], dist[i] = raschet(
            6, x[i], y[i], z[i], beaconsx, x_mayak[i - 1], y_mayak[i - 1], z[i]
        )

    rtsln_xy["x"][i] = x_mayak[i]
    rtsln_xy["y"][i] = y_mayak[i]
    distan["dist1"][i] = dist[i][0]
    distan["dist2"][i] = dist[i][1]
    distan["dist3"][i] = dist[i][2]
    distan["dist4"][i] = dist[i][3]
    distan["dist5"][i] = dist[i][4]
    distan["dist6"][i] = dist[i][5]

    rtsln["lat"][i], rtsln["lon"][i] = local_to_geographic(
        x_mayak[i], y_mayak[i]
    )
    logging.info(f"[{i} / {n_izm}]")

rtsln.to_csv("data/rtsln.csv", index=False)
distan.to_csv("data/distances.csv", index=False)
rtsln_xy.to_csv("data/rtsln_xy.csv", index=False)