import pandas as pd
import numpy as np

# Загрузка файлов
reference_trajectory = pd.read_csv('reference_trajectory.csv')
gnss = pd.DataFrame()

# Интерполяция столбца heading по времени
gnss['time'] = reference_trajectory['time']

# Добавление случайной погрешности величиной 1
gnss['heading'] = reference_trajectory['heading']

# Сохранение результата в новый CSV файл
gnss.to_csv('heading.csv', index=False)

print("Интерполяция и добавление погрешности завершены. Результат сохранен в 'gnss_with_heading.csv'.")
