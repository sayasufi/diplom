import pandas as pd
import numpy as np

# Загрузка файлов
reference_trajectory = pd.read_csv('reference_trajectory.csv')
gnss = pd.read_csv('gnss.csv')

# Интерполяция столбца heading по времени
gnss['heading'] = np.interp(gnss['time'], reference_trajectory['time'], reference_trajectory['heading'])

# Добавление случайной погрешности величиной 1
gnss['heading'] += np.random.uniform(-1, 1, size=gnss.shape[0])

# Сохранение результата в новый CSV файл
gnss.to_csv('gnss_with_heading.csv', index=False)

print("Интерполяция и добавление погрешности завершены. Результат сохранен в 'gnss_with_heading.csv'.")
