import numpy as np
distances = [100, 100000, 150, 100, 100, 100, 100]
n = 5
max_dis = max(distances)
min_dis = min(distances)



# Задаем измеренные расстояния до точки с погрешностями
distances = np.array(distances)
# Задаем веса для каждой дистанции маяка
koef = 1 / (n - 3)
weights_list = []
for i in distances:
    weights_list.append((i - min_dis) / (max_dis - min_dis) * (koef - 1) + 1)

print(weights_list)