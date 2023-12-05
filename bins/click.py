import math

import random
import numpy as np

from funcs import dist_least


def distanse_eq(x, y, x1, y1):
    return math.sqrt((x - x1) ** 2 + (y - y1) ** 2)


def raschet(n, x, y, beacon_coordinates):
    measured_distances = [0] * n

    for j in range(n):
        measured_distances[j] = distanse_eq(
            x, y, beacon_coordinates[j][0], beacon_coordinates[j][1]
        )

    Mean = 0
    Standard_deviation = 1
    dist_with_pogr = [0] * n

    for j in range(n):
        dist_with_pogr[j] = measured_distances[j] + (
            (
                random.uniform(5 / 1000 / 100, 15 / 1000 / 100)
                * measured_distances[j]
                + np.random.normal(
                    Mean, math.sqrt(Standard_deviation**2 + 0.5**2), 1
                )[0]
            )
        )

    x_dist, y_dist = dist_least(beacon_coordinates, dist_with_pogr)

    return x_dist, y_dist, dist_with_pogr
