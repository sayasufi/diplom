import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyins
import folium

gnss = pd.read_csv("/home/semyon/PycharmProjects/diplom/bins/data/rtsln.csv", index_col='time')
lat = gnss['lat']
lon = gnss['lon']
places = [[x[0], x[1]] for x in zip(lat, lon)]
m = folium.Map(places[0], tiles='OpenStreetMap', zoom_start=13)
polyline = folium.PolyLine(locations=places, color='blue', weight=3)
polyline.add_to(m)
m.save('rtsln.html')

# plt.plot(lon, lat)
# plt.show()