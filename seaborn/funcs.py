from pylab import *

from scipy.optimize import least_squares


def dist2(beacon_coords, distances):
    """Без весов"""

    def residuals(point):
        return np.linalg.norm(np.sqrt(np.sum((beacon_coords - point) ** 2, axis=1)) - distances)

    # Начальное предположение для координат точки
    initial_guess = np.array([0, 0])

    result = least_squares(residuals, initial_guess)

    # Координаты найденной точки
    return result.x


def dist_least(beacon_coords, distances):
    """С весами"""
    n = len(beacon_coords)
    max_dis = max(distances)
    min_dis = min(distances)

    # Задаем координаты маяков
    beacons = np.array(beacon_coords)

    # Задаем измеренные расстояния до точки с погрешностями
    distances = np.array(distances)
    # Задаем веса для каждой дистанции маяка
    koef = 1 / (n - 3)
    weights_list = []
    for i in distances:
        weights_list.append((i - min_dis) / (max_dis - min_dis) * (koef - 1) + 1)

    # weights_list = [1] * n

    weights = np.array(weights_list)

    # Функция, которую будем минимизировать
    def fun(point):
        return weights * (np.linalg.norm(beacons - point, axis=1) - distances)

    # Начальное приближение для координат точки
    initial_guess = np.array([0, 0])

    # Решаем задачу методом взвешенных наименьших квадратов
    result = least_squares(fun, initial_guess)

    # Получаем найденные координаты точки
    point_coordinates = result.x
    return point_coordinates
