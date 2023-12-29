import math

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def beacons(radius, n):
    n -= 1
    points = []
    step = 2 * math.pi / n
    angle = [0]
    points.append((radius * math.cos(angle[0]), radius * math.sin(angle[0])))
    for i in range(1, n):
        angle.append(angle[i - 1] + step)
        x = radius * math.cos(angle[i])
        y = radius * math.sin(angle[i])
        points.append((x, y))
    points.append((np.random.normal(0, 1, 1)[0], np.random.normal(0, 1, 1)[0]))
    return points


radius = 20000
kolvo_mayak = 6
beacon_coordinates = beacons(radius, kolvo_mayak)
b = np.array(beacon_coordinates)
x_coords = [point[0] for point in b]
y_coords = [point[1] for point in b]
print(x_coords, y_coords)
df = pd.read_excel("test222222.xlsx")

x = df["x"]
y = df["y"]
rmax = df["rmax"]
sko = df['sko']
mean = df['mean']


""""""
# df_rmax = pd.DataFrame({"x": x, "y": y, "r": rmax})
# pivot_df = df_rmax.pivot(index="y", columns="x", values="r")
# a = sns.color_palette("rocket_r", as_cmap=True)
# sns.heatmap(pivot_df, cmap=a)
# plt.gca().invert_yaxis()
# plt.grid()
# # Добавление большой зеленой точки
# for i in range(kolvo_mayak):
#     x = [x_coords[i] / 1500 + 10]
#     y = [y_coords[i] / 1500 + 10]
#
#     # Получение размеров данных и шагов на осях
#     x_ticks = plt.xticks()[0]
#     y_ticks = plt.yticks()[0]
#     x_step = x_ticks[1] - x_ticks[0]
#     y_step = y_ticks[1] - y_ticks[0]
#
#     # Вычисление координат точки
#     x_coord = x[0] * x_step + x_ticks[0]
#     y_coord = y[0] * y_step + y_ticks[0]
#
#     plt.scatter(x_coord, y_coord, c="green", s=100)
# plt.xlim(-1, 62)
# plt.ylim(-1, 62)
# plt.show()
""""""

""""""
df_mean = pd.DataFrame({"x": x, "y": y, "mean": mean})
pivot_df = df_mean.pivot(index="y", columns="x", values="mean")
a = sns.color_palette("rocket_r", as_cmap=True)
sns.heatmap(pivot_df, cmap=a)
plt.gca().invert_yaxis()
plt.grid()
for i in range(kolvo_mayak):
    x = [x_coords[i] / 1500 + 10]
    y = [y_coords[i] / 1500 + 10]

    # Получение размеров данных и шагов на осях
    x_ticks = plt.xticks()[0]
    y_ticks = plt.yticks()[0]
    x_step = x_ticks[1] - x_ticks[0]
    y_step = y_ticks[1] - y_ticks[0]

    # Вычисление координат точки
    x_coord = x[0] * x_step + x_ticks[0]
    y_coord = y[0] * y_step + y_ticks[0]

    plt.scatter(x_coord, y_coord, c="green", s=100)
plt.xlim(-1, 62)
plt.ylim(-1, 62)
plt.show()
""""""