import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def simulate_heading_errors(pitch_errors, roll_errors, random_scale=0.01):
    """
    Моделирует массив ошибок угла курса в зависимости от массивов ошибок углов тангажа и крена с учетом случайной составляющей.

    pitch_errors: массив ошибок угла тангажа в градусах (numpy array)
    roll_errors: массив ошибок угла крена в градусах (numpy array)
    random_scale: масштаб случайной составляющей

    Возвращает массив ошибок угла курса в градусах (numpy array).
    """
    # Нормализация ошибок тангажа и крена по их максимальному значению по модулю
    max_pitch_error = np.max(np.abs(pitch_errors))
    max_roll_error = np.max(np.abs(roll_errors))

    norm_pitch = pitch_errors / max_pitch_error
    norm_roll = roll_errors / max_roll_error

    # Основная ошибка курса, нормализованная в диапазоне [-1, 1]
    base_error = norm_pitch * norm_roll

    # Добавление случайной составляющей
    random_component = np.random.uniform(-random_scale, random_scale, size=base_error.shape)

    # Итоговая ошибка курса
    heading_errors = base_error + random_component
    heading_errors = np.clip(heading_errors, -1, 1)

    return heading_errors

# Пример использования
df = pd.read_csv('reference_trajectory.csv')
pitch_errors = df['pitch'].values
roll_errors = df['roll'].values

# Вычисление ошибок курса
heading_errors = simulate_heading_errors(pitch_errors, roll_errors)

df1 = pd.read_csv('heading.csv')
df1['heading'] += heading_errors
df1.to_csv('heading.csv')

# # Визуализация ошибок курса
# plt.figure(figsize=(10, 6))
# plt.plot(df['time'], heading_errors, label='Heading Error')
# plt.xlabel('Time')
# plt.ylabel('Heading Error (degrees)')
# plt.title('Heading Error over Time')
# plt.legend()
# plt.show()
