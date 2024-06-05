import numpy as np
import matplotlib.pyplot as plt


def simulate_heading_error(pitch_error_deg, roll_error_deg, random_scale=0.3):
    """
    Моделирует ошибку угла курса в зависимости от ошибки угла тангажа и крена с учетом случайной составляющей.

    pitch_error_deg: ошибка угла тангажа в градусах (-5 до 5)
    roll_error_deg: ошибка угла крена в градусах (-5 до 5)
    random_scale: масштаб случайной составляющей

    Возвращает ошибку угла курса в градусах (-1 до 1).
    """
    # Нормализуем ошибки тангажа и крена
    norm_pitch = pitch_error_deg / 5.0
    norm_roll = roll_error_deg / 5.0

    # Основная ошибка курса
    base_error = norm_pitch * norm_roll

    # Добавление случайной составляющей
    random_component = np.random.uniform(-random_scale, random_scale)

    # Итоговая ошибка курса
    heading_error_deg = base_error + random_component
    heading_error_deg = np.clip(heading_error_deg, -1, 1)

    return heading_error_deg


# Генерация данных для демонстрации
pitch_errors = np.linspace(-5, 5, 100)  # Ошибки тангажа от -5 до 5 градусов
roll_errors = np.linspace(-5, 5, 100)  # Ошибки крена от -5 до 5 градусов

# Массив для хранения ошибок курса
heading_errors = np.zeros((len(pitch_errors), len(roll_errors)))

for i, pitch_error in enumerate(pitch_errors):
    for j, roll_error in enumerate(roll_errors):
        heading_errors[i, j] = simulate_heading_error(pitch_error, roll_error)

# Визуализация ошибок курса
plt.figure(figsize=(10, 6))
plt.contourf(roll_errors, pitch_errors, heading_errors, 20, cmap='viridis')
plt.colorbar(label='Ошибка угла курса (градусы)')
plt.xlabel('Ошибка угла крена (градусы)')
plt.ylabel('Ошибка угла тангажа (градусы)')
plt.title('Ошибка угла курса в зависимости от ошибок углов тангажа и крена')
plt.show()
