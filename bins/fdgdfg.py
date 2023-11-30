import logging

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
        (58.01950503652435, 56.3459025115929),
        (57.99890367181495, 56.21822543477269),
        (57.97642599489462, 56.25844645809579),
        (57.99096964382019, 56.365263417805174),
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

gnss = pd.read_csv("data/gnss.csv")

lat = gnss["lat"]
lon = gnss["lon"]

n_izm = lat.shape[0]
x = [0] * n_izm
y = [0] * n_izm

for i in range(n_izm):
    x[i], y[i] = convert_geographic_to_projection(lat[i], lon[i])
    logging.info(f'Пересчет координат: {i} / {n_izm}')

x_mayak = [0] * n_izm
y_mayak = [0] * n_izm

for i in range(n_izm):
    x_mayak[i], y_mayak[i] = raschet(4, x[i], y[i], beacons_generate())
    logging.info(f'Расчет координат по маякам: {i} / {n_izm}')

lat_mayak = [0] * n_izm
lon_mayak = [0] * n_izm

for i in range(n_izm):
    lat_mayak[i], lon_mayak[i] = convert_projection_to_geographic(x_mayak[i], y_mayak[i])
    logging.info(f'Пересчет координат: {i} / {n_izm}')


places = [[x[0], x[1]] for x in zip(lat_mayak, lon_mayak)]
m = folium.Map(places[0], tiles='OpenStreetMap', zoom_start=13)
polyline = folium.PolyLine(locations=places, color='blue', weight=3)
polyline.add_to(m)
m.save('map_mayak.html')