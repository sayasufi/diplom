import numpy as np
from scipy.optimize import least_squares


def find_coordinates(beacons, distances):
    # Количество маяков
    num_beacons = len(beacons)

    # Функция для минимизации
    def fun(params):
        x, y = params[:2]
        # Вычисление расстояний от найденной точки до маяков
        predicted_distances = np.sqrt((beacons[:, 0] - x) ** 2 + (beacons[:, 1] - y) ** 2)
        # Вычисление взвешенной разницы между предсказанными и измеренными расстояниями
        diff = distances - predicted_distances
        return diff

    # Начальные значения координат точки
    initial_guess = np.zeros(2)

    # Минимизация функции с использованием метода наименьших квадратов
    result = least_squares(fun, initial_guess)

    # Возвращение найденных координат точки
    return result.x[:2]


# Пример данных
beacons = np.array([[1, 2], [3, 4], [5, 6]])
distances = np.array([2.5, 3.2, 4.8])

# Поиск координат точки
coordinates = find_coordinates(beacons, distances)
print("Координаты точки:", coordinates)
