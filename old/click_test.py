import math
import random

import matplotlib.pyplot as plt
import numpy as np
import pylab

from funcs import dist_least


def distanse_eq(x, y, x1, y1):
    return math.sqrt((x - x1) ** 2 + (y - y1) ** 2)


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


def onpick(event, n=None, beacon_coordinates=None):
    tb = plt.get_current_fig_manager().toolbar  # +gdf87g++
    if not tb.mode:  # +++
        m_x, m_y = event.x, event.y
        if 100 <= m_y <= 1000 and 1000 <= m_x <= 1700:
            ax = plt.gca()
            x, y = ax.transData.inverted().transform([m_x, m_y])
            raschet(n, x, y, beacon_coordinates)


def polet1(n, beacon_coordinates):
    r = beacon_coordinates[0][0]

    b = np.array(beacon_coordinates)
    x_coords = [point[0] for point in b]
    y_coords = [point[1] for point in b]

    plt.style.use('ggplot')
    pylab.figure(1, figsize=(100, 100))

    pylab.subplot(1, 2, 2)
    ax = plt.gca()
    fig = plt.figure(1)
    circle = plt.Circle((0, 0), r, fill=False, color='g')
    plt.plot(x_coords, y_coords, ".")
    plt.plot(0, 0, ".")
    plt.xlabel(f'Кол-во маяков = {n}')
    ax.add_patch(circle)
    plt.xlim(-15500, 15500)
    plt.ylim(-15500, 15500)
    pylab.title('Координаты точки')

    fig.canvas.mpl_connect('button_press_event',
                           lambda event: onpick(event, n=n, beacon_coordinates=beacon_coordinates))
    plt.show()


def raschet(n, x, y, beacon_coordinates):
    kolvo_izm = 500

    r = beacon_coordinates[0][0]
    b = np.array(beacon_coordinates)
    x_coords = [point[0] for point in b]
    y_coords = [point[1] for point in b]

    measured_distances = [[0 for _ in range(n)] for _ in range(kolvo_izm)]

    for k in range(kolvo_izm):
        for j in range(n):
            measured_distances[k][j] = distanse_eq(
                x, y, beacon_coordinates[j][0], beacon_coordinates[j][1]
            )

    Mean = 0
    Standard_deviation = 1
    dist_with_pogr = [[0 for _ in range(n)] for _ in range(kolvo_izm)]

    for k in range(kolvo_izm):
        for j in range(n):
            dist_with_pogr[k][j] = (
                    measured_distances[k][j]
                    + ((random.uniform(5 / 1000 / 100, 15 / 1000 / 100) * measured_distances[k][j] +
                        np.random.normal(Mean, math.sqrt(Standard_deviation ** 2 + 0.5 ** 2), 1)[0]
                        )))

    x_dist = [0 for _ in range(kolvo_izm)]
    y_dist = [0 for _ in range(kolvo_izm)]

    for k in range(kolvo_izm):
        # x_dist[k], y_dist[k] = calculate_distanse(beacon_coordinates, dist_with_pogr[k])
        x_dist[k], y_dist[k] = dist_least(beacon_coordinates, dist_with_pogr[k])
        # if k == 0:
        #     x_dist[k], y_dist[k] = dist2(beacon_coordinates, dist_with_pogr[k], 0, 0)
        # else:
        #     x_dist[k], y_dist[k] = dist2(beacon_coordinates, dist_with_pogr[k], x_dist[k - 1], y_dist[k - 1])

    pogrx = [0 for _ in range(kolvo_izm)]
    pogry = [0 for _ in range(kolvo_izm)]

    for k in range(kolvo_izm):
        pogrx[k] = x - x_dist[k]
        pogry[k] = y - y_dist[k]

    # sred_arifmx = sum(list(map(abs, pogrx))) / kolvo_izm
    # sred_arifmy = sum(list(map(abs, pogry))) / kolvo_izm
    # sred_geomx = gmean(list(map(abs, pogrx)))
    # sred_geomy = gmean(list(map(abs, pogry)))

    meanx = np.mean(pogrx)
    meany = np.mean(pogry)

    q1 = ((meanx - min(pogrx)) / 3 + (max(pogrx) - meanx) / 3) / 2
    q2 = ((meany - min(pogry)) / 3 + (max(pogry) - meany) / 3) / 2
    q = math.sqrt(q1 ** 2 + q2 ** 2)
    qmean = math.sqrt(meanx ** 2 + meany ** 2)

    pylab.clf()
    pylab.subplot(1, 2, 1)
    pylab.hist([pogrx, pogry], bins=20, histtype='step', label=('X', 'Y'))
    plt.xlabel(
        f'СКО = {Standard_deviation}\nМат ожидание = {round(qmean, 4)}\n{round(q, 4)}')
    pylab.legend(loc=2)
    pylab.title('Гистограмма погрешностей')

    pylab.subplot(1, 2, 2)
    ax = plt.gca()
    circle = plt.Circle((0, 0), r, fill=False, color='g')
    plt.plot(x_coords, y_coords, ".")
    plt.plot(x, y, ".", markersize=15)
    plt.xlabel(f'Кол-во маяков = {n}\nКоординаты точки: x = {round(x, 3)}, y = {round(y, 3)}')
    ax.add_patch(circle)
    plt.xlim(-15500, 15500)
    plt.ylim(-15500, 15500)
    pylab.title('Координаты точки')

    plt.show()


radius = 15000
kolvo_mayak = 13
beacon_coordinates = beacons(radius, kolvo_mayak)

polet1(kolvo_mayak, beacon_coordinates)
