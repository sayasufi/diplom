import pandas as pd
import numpy as np

# Чтение исходного файла
df = pd.read_csv('ref.csv')

# Создание нового DataFrame
df1 = pd.read_csv('rtsln.csv')
df1['alt'] = df['alt'] + np.random.uniform(0, 1, df1['time'].shape)
df1['VE'] = df['VE'] + np.random.uniform(0, 0.5, df1['time'].shape)
df1['VN'] = df['VN'] + np.random.uniform(0, 0.5, df1['time'].shape)
df1['VD'] = df['VD'] + np.random.uniform(0, 0.5, df1['time'].shape)



# Сохранение нового DataFrame в CSV файл
df1.to_csv('rtsln.csv', index=False)
