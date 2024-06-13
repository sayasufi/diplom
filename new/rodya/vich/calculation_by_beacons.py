import logging
import pandas as pd
import numpy as np
from click import raschet
from convert import geographic_to_local, local_to_geographic

# Настройка логгера
logger = logging.getLogger()
logger.setLevel(logging.NOTSET)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(logging.Formatter("%(asctime)s\t%(message)s"))
logger.addHandler(stream_handler)

def beacons_generate():
    beacons = (
        (55.306391, 38.591843, 150),
        (55.306762, 38.591443, 150),
        (55.307573, 38.596199, 150),
        (55.308038, 38.595845, 150),
        (55.309033, 38.601695, 150),
        (55.309458, 38.601335, 150),
    )

    beacons_xy = []
    for i, j, k in beacons:
        beacons_xy.append((*geographic_to_local(i, j), k))

    return beacons_xy

def moving_average(series, window_size):
    return series.rolling(window=window_size, center=True).mean()

# Чтение данных из файла
gnss = pd.read_csv("ref.csv")

lat = gnss["lat"]
lon = gnss["lon"]
alt = gnss["alt"]

beaconsx = beacons_generate()

n_izm = lat.shape[0]

x = [0] * n_izm
y = [0] * n_izm
z = [0] * n_izm

x_mayak = [0] * n_izm
y_mayak = [0] * n_izm
dist = [0] * n_izm

# Создание DataFrame с нужными столбцами
distan = pd.DataFrame()
rtsln_xy = pd.DataFrame(columns=["time", "x", "y"])
rtsln = pd.DataFrame(columns=["time", "lat", "lon"])

distan['time'] = gnss['time']
rtsln_xy['time'] = gnss['time']
rtsln['time'] = gnss['time']

# Основной расчет
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

    rtsln_xy.at[i, "x"] = x_mayak[i]
    rtsln_xy.at[i, "y"] = y_mayak[i]
    distan.at[i, "dist1"] = dist[i][0]
    distan.at[i, "dist2"] = dist[i][1]
    distan.at[i, "dist3"] = dist[i][2]
    distan.at[i, "dist4"] = dist[i][3]
    distan.at[i, "dist5"] = dist[i][4]
    distan.at[i, "dist6"] = dist[i][5]

    rtsln.at[i, "lat"], rtsln.at[i, "lon"] = local_to_geographic(
        x_mayak[i], y_mayak[i]
    )
    logging.info(f"[{i} / {n_izm}]")

# Применение скользящего среднего к каждому столбцу, кроме time
frequency = 50  # Гц
window_size = 5 * frequency  # Окно в 5 секунд

for column in distan.columns:
    if column != 'time':
        distan[column] = moving_average(distan[column], window_size).interpolate(method='linear').bfill().ffill()

for column in rtsln_xy.columns:
    if column != 'time':
        rtsln_xy[column] = moving_average(rtsln_xy[column], window_size).interpolate(method='linear').bfill().ffill()

for column in rtsln.columns:
    if column != 'time':
        rtsln[column] = moving_average(rtsln[column], window_size).interpolate(method='linear').bfill().ffill()

# Сохранение результатов
rtsln.to_csv("rtsln.csv", index=False)
distan.to_csv("distances.csv", index=False)
rtsln_xy.to_csv("rtsln_xy.csv", index=False)
