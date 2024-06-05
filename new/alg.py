import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from typing import Tuple, Optional

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Устанавливаем стиль для seaborn
sns.set(style="whitegrid")


class InertialNavigationSystem:
    def __init__(self, imu_file: str, reference_file: str, gnss_file: Optional[str] = None,
                 initial_position: Optional[Tuple[float, float, float]] = None,
                 initial_velocity: Optional[Tuple[float, float, float]] = None,
                 initial_orientation: Optional[Tuple[float, float, float]] = None):
        self.imu_file = imu_file
        self.gnss_file = gnss_file
        self.reference_file = reference_file
        self.imu_data, self.gnss_data, self.reference_data = self.read_data()
        self.num_samples = len(self.imu_data)
        self.dt = np.mean(np.diff(self.imu_data['time']))

        if initial_position is not None and initial_velocity is not None and initial_orientation is not None:
            self.state = self.initialize_state_with_initial_conditions(initial_position, initial_velocity,
                                                                       initial_orientation)
        else:
            self.state = self.initialize_state()

    def read_data(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
        """
        Чтение данных из CSV файлов.

        :return: Кортеж с данными IMU, GNSS (если присутствуют) и эталонными данными.
        """
        logging.info('Чтение данных из CSV файлов')
        imu_data = pd.read_csv(self.imu_file)
        reference_data = pd.read_csv(self.reference_file)
        gnss_data = None
        if self.gnss_file:
            gnss_data = pd.read_csv(self.gnss_file)
        logging.info('Данные успешно считаны')
        return imu_data, gnss_data, reference_data

    def initialize_state(self) -> dict:
        """
        Инициализация состояния INS.

        :return: Словарь с инициализированными состояниями.
        """
        logging.info('Инициализация состояния')
        state = {
            'position': np.zeros((self.num_samples, 3)),
            'velocity': np.zeros((self.num_samples, 3)),
            'orientation': np.zeros((self.num_samples, 4))
        }
        state['orientation'][0] = [0, 0, 0, 1]
        logging.info('Состояние инициализировано')
        return state

    def initialize_state_with_initial_conditions(self, initial_position: Tuple[float, float, float],
                                                 initial_velocity: Tuple[float, float, float],
                                                 initial_orientation: Tuple[float, float, float]) -> dict:
        """
        Инициализация состояния INS с заданными начальными условиями.

        :param initial_position: Начальная позиция (широта, долгота, высота).
        :param initial_velocity: Начальная скорость (VN, VE, VD).
        :param initial_orientation: Начальная ориентация (крена, тангажа, рыскания).
        :return: Словарь с инициализированными состояниями.
        """
        logging.info('Инициализация состояния с начальными условиями')
        state = {
            'position': np.zeros((self.num_samples, 3)),
            'velocity': np.zeros((self.num_samples, 3)),
            'orientation': np.zeros((self.num_samples, 4))
        }
        state['position'][0] = [initial_position[0] * 111320, initial_position[1] * 111320, initial_position[2]]
        state['velocity'][0] = initial_velocity
        state['orientation'][0] = R.from_euler('xyz', initial_orientation, degrees=True).as_quat()
        logging.info('Состояние инициализировано с начальными условиями')
        return state

    def integrate_imu_data(self):
        """
        Интеграция данных IMU для обновления состояния INS.
        """
        logging.info('Интеграция данных IMU')

        GRAVITY = 9.8175  # ускорение свободного падения в м/с²

        omegas = np.radians(self.imu_data[['gyro_x', 'gyro_y', 'gyro_z']].values) * self.dt
        accs = self.imu_data[['accel_x', 'accel_y', 'accel_z']].values

        # Коррекция ускорения по оси Z
        accs[:, 2] += GRAVITY

        positions = np.zeros((self.num_samples, 3))
        velocities = np.zeros((self.num_samples, 3))
        orientations = np.zeros((self.num_samples, 4))
        orientations[0] = self.state['orientation'][0]

        for i in range(1, self.num_samples):
            delta_rotation = R.from_rotvec(omegas[i - 1])
            orientation_prev = R.from_quat(orientations[i - 1])
            orientation_new = orientation_prev * delta_rotation
            orientations[i] = orientation_new.as_quat()

            acc_world = orientation_new.apply(accs[i - 1])
            velocities[i] = velocities[i - 1] + acc_world * self.dt
            positions[i] = positions[i - 1] + velocities[i] * self.dt

        self.state['position'] = positions
        self.state['velocity'] = velocities
        self.state['orientation'] = orientations
        logging.info('Интеграция данных IMU завершена')

    def correct_with_gnss(self, gnss_interp: pd.DataFrame):
        """
        Коррекция данных INS с использованием данных GNSS и магнитометра.

        :param gnss_interp: Интерполированные данные GNSS.
        """
        logging.info('Коррекция с использованием данных GNSS и магнитометра')
        gnss_positions = np.vstack([
            gnss_interp['lat'].values * 111320,  # Convert latitude to meters
            gnss_interp['lon'].values * 111320,  # Convert longitude to meters
            gnss_interp['alt'].values  # Altitude is already in meters
        ]).T
        gnss_velocities = gnss_interp[['VN', 'VE', 'VD']].values
        gnss_headings = np.radians(gnss_interp['heading'].values)  # Convert heading to radians

        self.state['position'][:len(gnss_positions)] = gnss_positions
        self.state['velocity'][:len(gnss_velocities)] = gnss_velocities

        # Correct orientations using heading
        for i in range(len(gnss_headings)):
            current_orientation = R.from_quat(self.state['orientation'][i])
            current_euler = current_orientation.as_euler('xyz')
            corrected_orientation = R.from_euler('xyz', [current_euler[0], current_euler[1], gnss_headings[i]])
            self.state['orientation'][i] = corrected_orientation.as_quat()

        logging.info('Коррекция с использованием данных GNSS и магнитометра завершена')

    def interpolate_reference_data(self) -> pd.DataFrame:
        """
        Интерполяция эталонных данных к временным меткам IMU.

        :return: Интерполированные эталонные данные.
        """
        logging.info('Интерполяция эталонных данных')
        reference_interp = pd.DataFrame()
        reference_interp['time'] = self.imu_data['time']
        reference_interp['lat'] = np.interp(self.imu_data['time'], self.reference_data['time'],
                                            self.reference_data['lat'])
        reference_interp['lon'] = np.interp(self.imu_data['time'], self.reference_data['time'],
                                            self.reference_data['lon'])
        reference_interp['alt'] = np.interp(self.imu_data['time'], self.reference_data['time'],
                                            self.reference_data['alt'])
        reference_interp['VN'] = np.interp(self.imu_data['time'], self.reference_data['time'],
                                           self.reference_data['VN'])
        reference_interp['VE'] = np.interp(self.imu_data['time'], self.reference_data['time'],
                                           self.reference_data['VE'])
        reference_interp['VD'] = np.interp(self.imu_data['time'], self.reference_data['time'],
                                           self.reference_data['VD'])
        reference_interp['roll'] = np.interp(self.imu_data['time'], self.reference_data['time'],
                                             self.reference_data['roll'])
        reference_interp['pitch'] = np.interp(self.imu_data['time'], self.reference_data['time'],
                                              self.reference_data['pitch'])
        reference_interp['heading'] = np.interp(self.imu_data['time'], self.reference_data['time'],
                                                self.reference_data['heading'])
        logging.info('Интерполяция эталонных данных завершена')
        return reference_interp

    @staticmethod
    def wrap_to_180(angles: np.ndarray) -> np.ndarray:
        """
        Приведение углов к диапазону [-180, 180] градусов.

        :param angles: Массив углов в градусах.
        :return: Массив углов, приведенных к диапазону [-180, 180].
        """
        return (angles + 180) % 360 - 180

    @staticmethod
    def calculate_yaw_error(estimated_yaw: np.ndarray, reference_yaw: np.ndarray) -> np.ndarray:
        """
        Вычисление ошибки по углу рыскания (Yaw) с учетом цикличности углов.

        :param estimated_yaw: Массив оцененных углов рыскания.
        :param reference_yaw: Массив эталонных углов рыскания.
        :return: Массив ошибок по углу рыскания.
        """
        error = estimated_yaw - reference_yaw
        error = (error + 180) % 360 - 180
        return error

    def calculate_errors(self, reference_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Расчет ошибок по позициям, скоростям и ориентациям.

        :param reference_data: Эталонные данные.
        :return: Кортеж с ошибками по позициям, скоростям и ориентациям.
        """
        logging.info('Расчет ошибок')
        reference_positions = np.vstack([
            reference_data['lat'].values * 111320,  # Convert latitude to meters
            reference_data['lon'].values * 111320,  # Convert longitude to meters
            reference_data['alt'].values  # Altitude is already in meters
        ]).T
        reference_velocities = reference_data[['VN', 'VE', 'VD']].values
        reference_orientations = R.from_euler('xyz', reference_data[['roll', 'pitch', 'heading']].values,
                                              degrees=True).as_quat()

        position_error = self.state['position'] - reference_positions
        velocity_error = self.state['velocity'] - reference_velocities

        estimated_orientations_euler = R.from_quat(self.state['orientation']).as_euler('xyz', degrees=True)
        reference_orientations_euler = R.from_quat(reference_orientations).as_euler('xyz', degrees=True)

        # Корректировка углов yaw для учета цикличности 360 градусов
        yaw_error = self.calculate_yaw_error(estimated_orientations_euler[:, 2], reference_orientations_euler[:, 2])

        orientation_error = np.vstack([
            estimated_orientations_euler[:, 0] - reference_orientations_euler[:, 0],
            estimated_orientations_euler[:, 1] - reference_orientations_euler[:, 1],
            yaw_error
        ]).T

        logging.info('Расчет ошибок завершен')
        return position_error, velocity_error, orientation_error

    def plot_errors(self, imu_time: np.ndarray, position_error: np.ndarray, velocity_error: np.ndarray,
                    orientation_error: np.ndarray,
                    start_time: Optional[float] = None, end_time: Optional[float] = None):
        """
        Построение графиков ошибок по позициям, скоростям и ориентациям на заданном промежутке времени.

        :param imu_time: Временные метки IMU.
        :param position_error: Ошибки по позициям.
        :param velocity_error: Ошибки по скоростям.
        :param orientation_error: Ошибки по ориентациям.
        :param start_time: Начало промежутка времени для построения графиков.
        :param end_time: Конец промежутка времени для построения графиков.
        """
        if start_time is not None and end_time is not None:
            mask = (imu_time >= start_time) & (imu_time <= end_time)
            imu_time = imu_time[mask]
            position_error = position_error[mask]
            velocity_error = velocity_error[mask]
            orientation_error = orientation_error[mask]

        # Plot errors for positions
        fig, axes = plt.subplots(3, 1, figsize=(12, 18))
        sns.lineplot(x=imu_time, y=position_error[:, 0], ax=axes[0])
        axes[0].set_title('Ошибка по X координате', fontsize=16)
        axes[0].set_xlabel('Время (с)', fontsize=14)
        axes[0].set_ylabel('Ошибка (м)', fontsize=14)
        axes[0].grid(True)

        sns.lineplot(x=imu_time, y=position_error[:, 1], ax=axes[1])
        axes[1].set_title('Ошибка по Y координате', fontsize=16)
        axes[1].set_xlabel('Время (с)', fontsize=14)
        axes[1].set_ylabel('Ошибка (м)', fontsize=14)
        axes[1].grid(True)

        sns.lineplot(x=imu_time, y=position_error[:, 2], ax=axes[2])
        axes[2].set_title('Ошибка по Z координате', fontsize=16)
        axes[2].set_xlabel('Время (с)', fontsize=14)
        axes[2].set_ylabel('Ошибка (м)', fontsize=14)
        axes[2].grid(True)

        plt.tight_layout()
        plt.show()

        # Plot errors for velocities
        fig, axes = plt.subplots(3, 1, figsize=(12, 18))
        sns.lineplot(x=imu_time, y=velocity_error[:, 0], ax=axes[0])
        axes[0].set_title('Ошибка по X скорости', fontsize=16)
        axes[0].set_xlabel('Время (с)', fontsize=14)
        axes[0].set_ylabel('Ошибка (м/с)', fontsize=14)
        axes[0].grid(True)

        sns.lineplot(x=imu_time, y=velocity_error[:, 1], ax=axes[1])
        axes[1].set_title('Ошибка по Y скорости', fontsize=16)
        axes[1].set_xlabel('Время (с)', fontsize=14)
        axes[1].set_ylabel('Ошибка (м/с)', fontsize=14)
        axes[1].grid(True)

        sns.lineplot(x=imu_time, y=velocity_error[:, 2], ax=axes[2])
        axes[2].set_title('Ошибка по Z скорости', fontsize=16)
        axes[2].set_xlabel('Время (с)', fontsize=14)
        axes[2].set_ylabel('Ошибка (м/с)', fontsize=14)
        axes[2].grid(True)

        plt.tight_layout()
        plt.show()

        # Plot errors for orientations
        fig, axes = plt.subplots(3, 1, figsize=(12, 18))
        sns.lineplot(x=imu_time, y=orientation_error[:, 0], ax=axes[0])
        axes[0].set_title('Ошибка по углу крена (Roll)', fontsize=16)
        axes[0].set_xlabel('Время (с)', fontsize=14)
        axes[0].set_ylabel('Ошибка (градусы)', fontsize=14)
        axes[0].grid(True)

        sns.lineplot(x=imu_time, y=orientation_error[:, 1], ax=axes[1])
        axes[1].set_title('Ошибка по углу тангажа (Pitch)', fontsize=16)
        axes[1].set_xlabel('Время (с)', fontsize=14)
        axes[1].set_ylabel('Ошибка (градусы)', fontsize=14)
        axes[1].grid(True)

        sns.lineplot(x=imu_time, y=orientation_error[:, 2], ax=axes[2])
        axes[2].set_title('Ошибка по углу рыскания (Yaw)', fontsize=16)
        axes[2].set_xlabel('Время (с)', fontsize=14)
        axes[2].set_ylabel('Ошибка (градусы)', fontsize=14)
        axes[2].grid(True)

        plt.tight_layout()
        plt.show()

    def run(self, start_time: Optional[float] = None, end_time: Optional[float] = None):
        """
        Основной метод для запуска всех вычислений и построения графиков.

        :param start_time: Начало промежутка времени для построения графиков.
        :param end_time: Конец промежутка времени для построения графиков.
        """
        self.integrate_imu_data()
        if self.gnss_file:
            gnss_interp = self.interpolate_gnss_data()
            self.correct_with_gnss(gnss_interp)
        reference_interp = self.interpolate_reference_data()
        position_error, velocity_error, orientation_error = self.calculate_errors(reference_interp)

        logging.info("Position Error: %s", position_error)
        logging.info("Velocity Error: %s", velocity_error)
        logging.info("Orientation Error: %s", orientation_error)

        self.plot_errors(self.imu_data['time'], position_error, velocity_error, orientation_error, start_time, end_time)


def run_without_correction(imu_file: str, reference_file: str, initial_position: Tuple[float, float, float],
                           initial_velocity: Tuple[float, float, float],
                           initial_orientation: Tuple[float, float, float],
                           start_time: Optional[float] = None, end_time: Optional[float] = None):
    """
    Запуск интеграции данных IMU без коррекции с использованием начальных условий.

    :param imu_file: Файл с данными IMU.
    :param reference_file: Файл с эталонными данными.
    :param initial_position: Начальная позиция (широта, долгота, высота).
    :param initial_velocity: Начальная скорость (VN, VE, VD).
    :param initial_orientation: Начальная ориентация (крена, тангажа, рыскания).
    :param start_time: Начало промежутка времени для построения графиков.
    :param end_time: Конец промежутка времени для построения графиков.
    """
    ins = InertialNavigationSystem(imu_file, reference_file,
                                   initial_position=initial_position,
                                   initial_velocity=initial_velocity,
                                   initial_orientation=initial_orientation)
    ins.run(start_time, end_time)


def main():
    imu_file = 'imu.csv'
    gnss_file = 'gnss.csv'
    reference_file = 'reference_trajectory.csv'

    # Запуск с коррекцией
    # ins = InertialNavigationSystem(imu_file, reference_file, gnss_file)
    start_time = 0
    end_time = 500
    # ins.run(start_time, end_time)

    # Запуск без коррекции
    # initial_position = (58.007384865, 56.325189914, 153.059)  # Начальные координаты (широта, долгота, высота)
    # initial_velocity = (0.0, 0.0, 0.0)  # Начальные скорости (VN, VE, VD)
    # initial_orientation = (-1.40, -1.36, 57.64)  # Начальные углы ориентации (крена, тангажа, рыскания)
    # run_without_correction(imu_file, reference_file, initial_position, initial_velocity, initial_orientation,
    #                        start_time, end_time)


if __name__ == "__main__":
    main()
