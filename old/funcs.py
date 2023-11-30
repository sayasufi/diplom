import numpy as np
import math

import scipy
from pylab import *
from scipy.stats import gmean
from scipy.optimize import least_squares


def transpose_matrix(matrix):
    rows, cols = matrix.shape
    transposed_matrix = np.zeros((cols, rows))
    for i in range(rows):
        for j in range(cols):
            transposed_matrix[j, i] = matrix[i, j]
    return transposed_matrix


def multiply_matrices(A, B):
    # Проверка размеров матриц
    if A.shape[1] != B.shape[0]:
        raise ValueError("Размеры матриц несовместимы для умножения!")

    # Создание результирующей матрицы
    result = np.zeros((A.shape[0], B.shape[1]))

    # Умножение матриц
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                result[i, j] += A[i, k] * B[k, j]

    return result


def solve_linear_equations(A, b):
    # Проверка размерности матрицы A
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Матрица A должна быть квадратной")

    # Проверка размерности матриц A и b
    if A.shape[0] != b.shape[0]:
        raise ValueError(
            "Размерность матрицы A должна быть совместима с размерностью матрицы b"
        )

    # Размерность системы уравнений
    n = A.shape[0]

    # Создание расширенной матрицы [A | b]
    augmented_matrix = np.column_stack((A, b))

    # Прямой ход метода Гаусса
    for i in range(n):
        # Поиск максимального элемента в столбце i
        max_index = np.argmax(np.abs(augmented_matrix[i:, i])) + i

        # Перестановка строк
        augmented_matrix[[i, max_index]] = augmented_matrix[[max_index, i]]

        # Нормализация текущей строки
        augmented_matrix[i, :] /= augmented_matrix[i, i]

        # Вычитание текущей строки из остальных строк
        for j in range(i + 1, n):
            augmented_matrix[j, :] -= augmented_matrix[j, i] * augmented_matrix[i, :]

    # Обратный ход метода Гаусса
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = augmented_matrix[i, -1] - np.dot(augmented_matrix[i, :-1], x)

    return x


def calculate_azimut(azimuth_angles, beacon_coordinates):
    num_beacons = len(beacon_coordinates)

    # Создаем матрицу коэф. параметрических уравнений поправок A и вектор свободных членов парам. ур-й b для решения через МНК
    A = np.zeros((len(azimuth_angles), 2))
    b = np.zeros((len(azimuth_angles), 1))

    for i, angle in enumerate(azimuth_angles):
        x_i, y_i = beacon_coordinates[i]
        A[i, 0] = np.cos(np.radians(angle))
        A[i, 1] = -np.sin(np.radians(angle))
        b[i] = x_i * np.cos(np.radians(angle)) - y_i * np.sin(np.radians(angle))

    # Решаем систему уравнений через МНК
    A_transpose = transpose_matrix(A)
    A_transpose_A = multiply_matrices(A_transpose, A)  # матрица коэф. нормальных ур-й
    A_transpose_b = multiply_matrices(
        A_transpose, b
    )  # вектор свободных членов нормальных ур-й
    least_squares_solution = solve_linear_equations(A_transpose_A, A_transpose_b)

    # Вычисление координат точки
    xa = least_squares_solution[0]
    ya = least_squares_solution[1]

    return xa, ya


def calculate_distanse(beacon_coordinates, measured_distances):
    num_beacons = len(beacon_coordinates)

    # Создаем матрицу коэф. параметрических уравнений поправок A и вектор свободных членов парам. ур-й b для решения через МНК
    Ad = np.zeros((num_beacons, 2))
    bd = np.zeros((num_beacons, 1))

    for i in range(num_beacons):
        Ad[i][0] = 2 * (beacon_coordinates[i][0] - beacon_coordinates[0][0])
        Ad[i][1] = 2 * (beacon_coordinates[i][1] - beacon_coordinates[0][1])
        bd[i] = (
                (beacon_coordinates[i][0] ** 2 - beacon_coordinates[0][0] ** 2)
                + (beacon_coordinates[i][1] ** 2 - beacon_coordinates[0][1] ** 2)
                - (measured_distances[i] ** 2 - measured_distances[0] ** 2)
        )

    # Решение системы уравнений методом наименьших квадратов
    Ad_transpose = transpose_matrix(Ad)
    print(Ad)
    Ad_transpose_A = multiply_matrices(
        Ad_transpose, Ad
    )  # матрица коэф. нормальных ур-й
    Ad_transpose_b = multiply_matrices(
        Ad_transpose, bd
    )  # вектор свободных членов нормальных ур-й
    least_squares_solution = solve_linear_equations(Ad_transpose_A, Ad_transpose_b)

    # Вычисление координат точки
    xd = least_squares_solution[0]
    yd = least_squares_solution[1]

    return xd, yd


def find_inverse_matrix(matrix):
    try:
        inverse_matrix = np.linalg.inv(matrix)
        return inverse_matrix
    except np.linalg.LinAlgError:
        print("Матрица не имеет обратной")
        return None


def extrapolation(dist_list_with_pogr, n):
    time_sec = len(dist_list_with_pogr)
    count_mayak = len(dist_list_with_pogr[0])
    extr_dist = [[0 for _ in range(count_mayak)] for _ in range(time_sec)]

    for i in range(count_mayak):
        for j in range(0, n // 2):
            extr_dist[j][i] = dist_list_with_pogr[j][i]

    for i in range(count_mayak):
        for j in range(n // 2, time_sec - n // 2):
            a = [dist_list_with_pogr[k][i] for k in range(j - n // 2, j + n // 2)]
            extr_dist[j][i] = sum(a) / len(a)

    for i in range(count_mayak):
        for j in range(time_sec - n // 2, time_sec):
            extr_dist[j][i] = dist_list_with_pogr[j][i]

    return extr_dist


def extrapolation_azimut(dist_list_with_pogr, n):
    time_sec = len(dist_list_with_pogr)
    count_mayak = len(dist_list_with_pogr[0])
    extr_dist = [[0 for _ in range(count_mayak)] for _ in range(time_sec)]

    for i in range(count_mayak):
        for j in range(0, n // 2):
            extr_dist[j][i] = dist_list_with_pogr[j][i]

    for i in range(count_mayak):
        for j in range(n // 2, time_sec - n // 2):
            a = [dist_list_with_pogr[k][i] for k in range(j - n // 2, j + n // 2)]
            extr_dist[j][i] = sum(a) / len(a)

    for i in range(count_mayak):
        for j in range(time_sec - n // 2, time_sec):
            extr_dist[j][i] = dist_list_with_pogr[j][i]

    return extr_dist


def dist2(beacon_coords, distances):
    def residuals(point):
        return np.linalg.norm(np.sqrt(np.sum((beacon_coords - point) ** 2, axis=1)) - distances)

    # Начальное предположение для координат точки
    initial_guess = np.array([0, 0])

    result = least_squares(residuals, initial_guess)

    # Координаты найденной точки
    return result.x


def dist_least(beacon_coords, distances):
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
