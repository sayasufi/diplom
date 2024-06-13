import pandas as pd
import numpy as np

# Чтение исходного файла
df = pd.read_csv('reference_trajectory.csv')

# Создание нового DataFrame
df1 = pd.DataFrame()
df1['time'] = df['time']
df1['VE'] = df['VE']
df1['VN'] = df['VN']

# Функция для осреднения значений в окне из 20 значений
def average_in_chunks(data, chunk_size):
    return data.groupby(data.index // chunk_size).mean()

# Применение функции для уменьшения количества значений в 20 раз
chunk_size = 20
df_reduced = df1.apply(lambda x: average_in_chunks(x, chunk_size))

# Добавление случайной погрешности в диапазоне -0.05 до +0.05
np.random.seed(0)  # Для воспроизводимости результатов
random_noise = np.random.uniform(0, 0.5, df_reduced.shape)
df_reduced += random_noise

# Сохранение нового DataFrame в CSV файл
df_reduced.to_csv('odometr.csv', index=False)
