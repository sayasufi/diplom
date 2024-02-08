"""
Обработка реальных данных
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyins

plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.grid'] = True
plt.rcParams['font.size'] = 12

"""
В этом примере мы обработаем данные, записанные с реальных датчиков IMU и GNSS.

Настройка и выполнение фильтра
Загрузите эталонную траекторию, данные IMU и GNSS.

Эталонная траектория была вычислена с использованием расширенного 
алгоритма постобработки с поправками RTK, одометром и т.д.
IMU заранее компенсировал значительные отклонения, ошибки масштаба и несоосности.

Данные GNSS предоставляются в базовой форме - только данные о 
местоположении и скорости, никаких оценок точности.
Все данные синхронизированы с общими часами.
"""

reference_trajectory = pd.read_csv("data/rodya/vzlet_ref.csv", index_col='time')
imu = pd.read_csv("data/rodya/vzlet_imu.csv", index_col='time')
gnss = pd.read_csv("data/rodya/vzlet_sns_150.csv", index_col='time')

"""
Мы выбираем приблизительно подходящие параметры для определения 
местоположения и скорости GNSS с точностью 1 м и 0,3 м/с.

Обратите внимание, что независимые случайные ошибки обычно не могут адекватно 
учитывать медленно изменяющиеся смещения положения GNSS из-за некомпенсированных ошибок. 
Но при слабосвязанной интеграции INS/GNSS в этом отношении мало что можно сделать.

Известно, как антенна GNSS расположена относительно IMU.
"""

imu_to_ant_b = np.array([0.000, 0.000, 0.000])
position_meas = pyins.measurements.Position(gnss, 1.0, imu_to_ant_b)
velocity_meas = pyins.measurements.NedVelocity(gnss, 0.3, imu_to_ant_b)

"""
Вычислите приращения из необработанных значений IMU.
"""

increments = pyins.strapdown.compute_increments_from_imu(imu, 'rate')

"""
Начальное положение-скорость-ориентация вычисляются по 
эталонному pva путем сложения ошибок со следующими стандартными отклонениями:
"""

position_sd = 5.0
velocity_sd = 0.5
level_sd = 0.2
azimuth_sd = 1.0

pva_error = pyins.sim.generate_pva_error(position_sd, velocity_sd, level_sd, azimuth_sd, 0)
pva_initial = pyins.sim.perturb_pva(reference_trajectory.iloc[0], pva_error)

"""
Используемый IMU представляет собой MEMS-датчик низкого или среднего класса. 
Для его моделирования используются следующие параметры. 
Случайные отклонения смещения несколько увеличены, чтобы учесть возможные колебания температуры.
"""

gyro_model = pyins.inertial_sensor.EstimationModel(
    bias_sd=300.0 * pyins.transform.DH_TO_RS,
    noise=1.0 * pyins.transform.DRH_TO_RRS,
    bias_walk=30.0 * pyins.transform.DH_TO_RS / 60)

accel_model = pyins.inertial_sensor.EstimationModel(
    bias_sd=0.05,
    noise=0.1 / 60,
    bias_walk=0.01 / 60)

"""
Теперь все готово к запуску фильтра.
"""

result = pyins.filters.run_feedback_filter(
    pva_initial, position_sd, velocity_sd, level_sd, azimuth_sd,
    increments, gyro_model, accel_model,
    measurements=[position_meas, velocity_meas])

"""
Анализ результатов
Сначала давайте построим график нормализованных инноваций в измерениях.
"""

plt.plot(result.innovations['Position'], label=['lat', 'lon', 'alt'])
plt.xlabel("System time, s")
plt.legend()
plt.savefig('png/2/1.png', dpi=600)
plt.show()
plt.clf()

"""
Здесь мы видим, что количество нововведений не превышает 1 (что хорошо), 
но совсем не похоже на последовательность белого цвета 
(что ожидаемо, если выполняются предположения фильтра Калмана).

К сожалению, это типичная ситуация для слабосвязанной интеграции INS/GNSS. 
Местоположение GNSS вычисляется с помощью сложных алгоритмов (обычно включающих фильтрацию Калмана), 
и его ошибки непредсказуемым образом коррелируются во времени.
"""

plt.plot(result.innovations['NedVelocity'], label=['VN', 'VE', 'VD'])
plt.legend()
plt.tight_layout()
plt.savefig('png/2/2.png', dpi=600)
plt.show()
plt.clf()

"""
Почти то же самое можно сказать и об инновациях velocity.
Обычно ошибки скорости GNSS ведут себя скорее как некоррелированная последовательность,
но они более подвержены алгоритмическим задержкам из-за процесса фильтрации.
Здесь также вероятно, что мы наблюдаем некоторые неоптимальности фильтра pyins.

В целом, картина инноваций не выглядит ужасной для упрощенного фильтра pyins и внешних данных GNSS.

Теперь давайте вычислим и построим график ошибок (относительно базовой траектории) 
и границ в 3 сигмы, используя стандартное отклонение фильтра.
"""

trajectory_error = pyins.transform.compute_state_difference(result.trajectory, reference_trajectory)

for i, col in enumerate(trajectory_error.columns, start=1):
    plt.plot(trajectory_error[col])
    plt.tight_layout()
    plt.savefig(f'png/2/3{i}.png', dpi=600)
    plt.show()
    plt.clf()

"""
Ошибки определения местоположения здесь не представляют большого интереса, 
поскольку решения о местоположении, по существу, вычисляются для разных систем 
отсчета из-за смещений положения GNSS (опорное положение RTK также может быть 
смещено от истинного кадра из-за смещения координат базовой станции). 
Здесь это особенно очевидно для ошибки высоты (снижения).

Об ошибках скорости можно сказать немногое, они адекватны.

Ошибки угла ориентации выглядят не особенно большими - они касаются границ в 3 сигмы, 
и есть среднее смещение для углов крена и тангажа. 
Но в целом они адекватны для упрощенных слабосвязанных INS/GNSS и укладываются в границы в 3 сигмы.

Давайте также построим график оценки параметров гироскопа и акселерометра, 
чтобы убедиться в адекватности стандартного отклонения начального смещения.
"""

plt.plot(result.gyro * pyins.transform.RS_TO_DH, label=['bias_x', 'bias_y', 'bias_z'])
plt.legend()
plt.xlabel("System time, s")
plt.tight_layout()
plt.savefig('png/2/4.png', dpi=600)
plt.show()
plt.clf()

plt.plot(result.accel, label=['bias_x', 'bias_y', 'bias_z'])
plt.legend()
plt.xlabel("System time, s")
plt.tight_layout()
plt.savefig('png/2/5.png', dpi=600)
plt.show()
plt.clf()

"""
Все выглядит разумно. Имеются некоторые ложные изменения оценок, 
скорее всего, из-за проблемных статистических свойств данных о местоположении и скорости GNSS.
"""
