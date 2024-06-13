import folium
import pandas as pd
import numpy as np

# Чтение данных
gnss = pd.read_csv("rtsln.csv", index_col='time')

# Преобразование координат
lat = gnss['lat']
lon = gnss['lon']
places = [[x[0], x[1]] for x in zip(lat, lon)]

# Дополнительные точки
additional_points = [
    (55.306391, 38.591843, 150),
    (55.306762, 38.591443, 150),
    (55.307573, 38.596199, 150),
    (55.308038, 38.595845, 150),
    (55.309033, 38.601695, 150),
    (55.309458, 38.601335, 150),
]

# Создание карты с гибридными плитками
m = folium.Map(places[0], zoom_start=13)

# Добавление гибридных плиток (например, Google Maps)
folium.TileLayer(
    tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
    attr='Google',
    name='Google Satellite Hybrid',
    overlay=True,
    control=True
).add_to(m)

# Добавление маршрута
polyline = folium.PolyLine(locations=places, color='blue', weight=3)
polyline.add_to(m)

# Добавление дополнительных точек
for point in additional_points:
    folium.Marker(
        location=[point[0], point[1]],
        popup=f'Altitude: {point[2]} m',
        icon=folium.Icon(color='red')
    ).add_to(m)


# Функция для добавления сетки на карту
def add_grid(map_obj, center_lat, center_lon, interval_km, radius_km):
    # Преобразование интервала из км в градусы
    interval_lat_deg = interval_km / 111  # 1 градус широты примерно равен 111 км
    interval_lon_deg = interval_km / (111 * np.cos(np.radians(center_lat)))  # корректировка для долготы

    # Преобразование радиуса из км в градусы
    radius_lat_deg = radius_km / 111
    radius_lon_deg = radius_km / (111 * np.cos(np.radians(center_lat)))

    start_lat = center_lat - radius_lat_deg
    end_lat = center_lat + radius_lat_deg
    start_lon = center_lon - radius_lon_deg
    end_lon = center_lon + radius_lon_deg

    lat_lines = np.arange(start_lat, end_lat, interval_lat_deg)
    lon_lines = np.arange(start_lon, end_lon, interval_lon_deg)

    for lat in lat_lines:
        folium.PolyLine([(lat, start_lon), (lat, end_lon)], color='white', weight=1, opacity=0.8,
                        dash_array='5,5').add_to(map_obj)

    for lon in lon_lines:
        folium.PolyLine([(start_lat, lon), (end_lat, lon)], color='white', weight=1, opacity=0.8,
                        dash_array='5,5').add_to(map_obj)


# Добавление сетки на карту с интервалом 1 км и радиусом 100 км
add_grid(m, places[0][0], places[0][1], 1, 100)

# Сохранение карты в файл
m.save('maps_with_large_grid.html')
