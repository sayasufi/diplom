import math
import time
from random import randrange

from pylab import *
from scipy.stats import gmean

from funcs import calculate_azimut, calculate_distanse


def distanse_eq(x, y, x1, y1):
    return sqrt((x - x1) ** 2 + (y - y1) ** 2)


def azimut_eq(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    angle = math.atan2(dx, dy)
    angle_deg = math.degrees(angle)
    if angle_deg < 0:
        angle_deg += 360
    return angle_deg


def beacons(n):
    return [(randrange(-10000, 10000, 100), randrange(-10000, 10000, 100)) for _ in range(n)]


def polet(n, seconds_polet, beacon_coordinates):
    time_start = time.perf_counter()
    x = [0 for _ in range(seconds_polet)]
    y = [0 for _ in range(seconds_polet)]
    for i in range(0, seconds_polet):
        x[i] = 7000 * cos(i / 180 * pi / seconds_polet * 360)
        y[i] = 7000 * sin(i / 180 * pi / seconds_polet * 360)

    measured_distances = [[0 for _ in range(n)] for _ in range(seconds_polet)]
    azimuth_angles = [[0 for _ in range(n)] for _ in range(seconds_polet)]

    for i in range(seconds_polet):
        for j in range(n):
            measured_distances[i][j] = distanse_eq(
                x[i], y[i], beacon_coordinates[j][0], beacon_coordinates[j][1]
            )
            azimuth_angles[i][j] = azimut_eq(
                beacon_coordinates[j][0], beacon_coordinates[j][1], x[i], y[i]
            )

    dist_with_pogr = [[0 for _ in range(n)] for _ in range(seconds_polet)]
    azimuth_with_pogr = [[0 for _ in range(n)] for _ in range(seconds_polet)]
    for i in range(seconds_polet):
        for j in range(n):
            dist_with_pogr[i][j] = (
                    measured_distances[i][j]
                    + randrange(-100, 100, 1) * measured_distances[i][j] / 1000 / 100
            )  # погрешность при дисстанции 10км 10м

            azimuth_with_pogr[i][j] = (
                    azimuth_angles[i][j] + randrange(-100, 100, 1) / 1000
            )  # погрешность 0.1 градус

    x_dist = [0 for _ in range(seconds_polet)]
    y_dist = [0 for _ in range(seconds_polet)]
    x_azimut = [0 for _ in range(seconds_polet)]
    y_azimut = [0 for _ in range(seconds_polet)]

    for i in range(seconds_polet):
        x_dist[i], y_dist[i] = calculate_distanse(beacon_coordinates, dist_with_pogr[i])
        x_azimut[i], y_azimut[i] = calculate_azimut(
            azimuth_with_pogr[i], beacon_coordinates
        )

    # plt.figure()
    # plt.plot(x, y)
    # plt.plot(x_dist, y_dist)
    # plt.plot(x_azimut, y_azimut)
    # plt.show()

    dist_pogr = [0 for _ in range(seconds_polet)]
    azimut_pogr = [0 for _ in range(seconds_polet)]
    for i in range(seconds_polet):
        dist_pogr[i] = sqrt((x_dist[i] - x[i]) ** 2 + (y_dist[i] - y[i]) ** 2)
        azimut_pogr[i] = sqrt((x_azimut[i] - x[i]) ** 2 + (y_azimut[i] - y[i]) ** 2)
    print(f'Кол-во маяков = {n}\nВремя полета = {seconds_polet} сек.')
    print(*beacon_coordinates, sep='\n', end='\n=========================\n')
    print(f"Средняя арифм поргешность по дистанциям = {sum(dist_pogr) / len(dist_pogr)}")
    print(f"Средняя геом поргешность по дистанциям = {gmean(dist_pogr)}")
    print(f"Средняя арифм поргешность по углам = {sum(azimut_pogr) / len(azimut_pogr)}")
    print(f'Время выполнения функции = {time.perf_counter() - time_start} с.')
    print(f'=========================\n')


a = beacons(40)
# for i in range(3, 16, 2):
#     polet(i, 4400, a[:i])

polet(10, 4400, a[:10])
# polet(4, 1000, a[:4])
