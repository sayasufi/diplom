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
        (58.050575, 56.322439, 150),
        (57.992717, 56.455636, 150),
        (57.927987, 56.343635, 150),
        (57.985291, 56.207580, 150),
        (57.991502, 56.334067, 150),
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

n_izm = lat.shape[0] - 1

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
            5, x[i], y[i], z[i], beaconsx, 0, 0, z[i]
        )
    else:
        x_mayak[i], y_mayak[i], dist[i] = raschet(
            5, x[i], y[i], z[i], beaconsx, x_mayak[i - 1], y_mayak[i - 1], z[i]
        )

    rtsln_xy["x"][i] = x_mayak[i]
    rtsln_xy["y"][i] = y_mayak[i]
    distan["dist1"][i] = dist[i][0]
    distan["dist2"][i] = dist[i][1]
    distan["dist3"][i] = dist[i][2]
    distan["dist4"][i] = dist[i][3]
    distan["dist5"][i] = dist[i][4]

    rtsln["lat"][i], rtsln["lon"][i] = local_to_geographic(
        x_mayak[i], y_mayak[i]
    )
    logging.info(f"[{i} / {n_izm}]")


rtsln.to_csv("data/rtsln.csv", index=False)
distan.to_csv("data/distances.csv", index=False)
rtsln_xy.to_csv("data/rtsln_xy.csv", index=False)