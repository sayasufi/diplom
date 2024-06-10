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


def geographic_to_local(lat, lon, origin_lat, origin_lon):
    R_earth = 6378137  # радиус Земли в метрах
    dlat = np.radians(lat - origin_lat)
    dlon = np.radians(lon - origin_lon)
    x = R_earth * dlon * np.cos(np.radians(origin_lat))
    y = R_earth * dlat
    return x, y


def local_to_geographic(x, y, origin_lat, origin_lon):
    R_earth = 6378137  # радиус Земли в метрах
    dlat = y / R_earth
    dlon = x / (R_earth * np.cos(np.radians(origin_lat)))
    lat = np.degrees(dlat) + origin_lat
    lon = np.degrees(dlon) + origin_lon
    return lat, lon


class BaseINS:
    def __init__(self, imu_file: str, reference_file: str, heading_file: Optional[str] = None):
        self.imu_file = imu_file
        self.reference_file = reference_file
        self.heading_file = heading_file
        self.imu_data, self.reference_data, self.heading_data = self.read_data()
        self.num_samples = len(self.imu_data)
        self.dt = np.mean(np.diff(self.imu_data['time']))
        self.origin_lat = self.reference_data['lat'].iloc[0]
        self.origin_lon = self.reference_data['lon'].iloc[0]
        self.state = self.initialize_state()

    def read_data(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Чтение данных из CSV файлов.

        :return: Кортеж с данными IMU, эталонными данными и данными магнитометра.
        """
        logging.info('Чтение данных из CSV файлов')
        imu_data = pd.read_csv(self.imu_file)
        reference_data = pd.read_csv(self.reference_file)
        heading_data = None
        if self.heading_file:
            heading_data = pd.read_csv(self.heading_file)
        logging.info('Данные успешно считаны')
        return imu_data, reference_data, heading_data

    def initialize_state(self) -> dict:
        """
        Инициализация состояния INS с начальными данными из эталонной траектории.

        :return: Словарь с инициализированными состояниями.
        """
        logging.info('Инициализация состояния с начальными данными из эталонной траектории')
        initial_position = (
            self.reference_data['lat'].iloc[0], self.reference_data['lon'].iloc[0], self.reference_data['alt'].iloc[0])
        initial_velocity = (
            self.reference_data['VN'].iloc[0], self.reference_data['VE'].iloc[0], self.reference_data['VD'].iloc[0])
        initial_orientation = (self.reference_data['roll'].iloc[0], self.reference_data['pitch'].iloc[0],
                               self.reference_data['heading'].iloc[0])

        state = {
            'position': np.zeros((self.num_samples, 3)),
            'velocity': np.zeros((self.num_samples, 3)),
            'orientation': np.zeros((self.num_samples, 4))
        }
        x, y = geographic_to_local(initial_position[0], initial_position[1], self.origin_lat, self.origin_lon)
        state['position'][0] = [x, y, initial_position[2]]
        state['velocity'][0] = initial_velocity
        state['orientation'][0] = R.from_euler('xyz', initial_orientation, degrees=True).as_quat()
        logging.info('Состояние инициализировано с начальными данными из эталонной траектории')
        return state

    def calculate_errors(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Расчет ошибок по позициям, скоростям и ориентациям.

        :return: Кортеж с ошибками по позициям, скоростям и ориентациям.
        """
        logging.info('Расчет ошибок')

        reference_positions = np.vstack([
            geographic_to_local(self.reference_data['lat'].values, self.reference_data['lon'].values, self.origin_lat,
                                self.origin_lon)[0],
            geographic_to_local(self.reference_data['lat'].values, self.reference_data['lon'].values, self.origin_lat,
                                self.origin_lon)[1],
            self.reference_data['alt'].values  # Altitude is already in meters
        ]).T
        reference_velocities = self.reference_data[['VN', 'VE', 'VD']].values
        reference_orientations = R.from_euler('xyz', self.reference_data[['roll', 'pitch', 'heading']].values,
                                              degrees=True).as_quat()

        min_len = min(len(self.state['position']), len(reference_positions))
        position_error = self.state['position'][:min_len] - reference_positions[:min_len]
        velocity_error = self.state['velocity'][:min_len] - reference_velocities[:min_len]

        estimated_orientations_euler = R.from_quat(self.state['orientation'][:min_len]).as_euler('xyz', degrees=True)
        reference_orientations_euler = R.from_quat(reference_orientations[:min_len]).as_euler('xyz', degrees=True)

        orientation_error = np.vstack([
            estimated_orientations_euler[:, 0] - reference_orientations_euler[:, 0],
            estimated_orientations_euler[:, 1] - reference_orientations_euler[:, 1],
            estimated_orientations_euler[:, 2] - reference_orientations_euler[:, 2]
        ]).T

        logging.info('Расчет ошибок завершен')
        return position_error, velocity_error, orientation_error

    def plot_errors(self, imu_time: np.ndarray, position_error: np.ndarray, velocity_error: np.ndarray,
                    orientation_error: np.ndarray, start_time: Optional[float] = None, end_time: Optional[float] = None,
                    window: int = 5):
        """
        Построение графиков ошибок по позициям, скоростям и ориентациям на заданном промежутке времени.

        :param imu_time: Временные метки IMU.
        :param position_error: Ошибки по позициям.
        :param velocity_error: Ошибки по скоростям.
        :param orientation_error: Ошибки по ориентациям.
        :param start_time: Начало промежутка времени для построения графиков.
        :param end_time: Конец промежутка времени для построения графиков.
        :param window: Размер окна для сглаживания данных в секундах.
        """
        if start_time is not None and end_time is not None:
            mask = (imu_time >= start_time) & (imu_time <= end_time)
            imu_time = imu_time[mask]
            position_error = position_error[mask]
            velocity_error = velocity_error[mask]
            orientation_error = orientation_error[mask]

        # Определение размера окна в индексах
        window_size = int(window / self.dt)

        # Применение скользящего среднего для ошибок
        position_error_smooth = pd.DataFrame(position_error).rolling(window=window_size, min_periods=1,
                                                                     center=True).mean().values
        velocity_error_smooth = pd.DataFrame(velocity_error).rolling(window=window_size, min_periods=1,
                                                                     center=True).mean().values
        orientation_error_smooth = pd.DataFrame(orientation_error).rolling(window=window_size, min_periods=1,
                                                                           center=True).mean().values

        fig, axes = plt.subplots(3, 1, figsize=(12, 18))
        sns.lineplot(x=self.state['position'][0], y=self.state['position'][1], ax=axes[0])
        axes[0].set_title('Ошибка по X координате', fontsize=16)
        axes[0].set_xlabel('Время (с)', fontsize=14)
        axes[0].set_ylabel('Ошибка (м)', fontsize=14)
        axes[0].grid(True)

        # Plot errors for positions
        fig, axes = plt.subplots(3, 1, figsize=(12, 18))
        sns.lineplot(x=imu_time, y=position_error_smooth[:, 0], ax=axes[0])
        axes[0].set_title('Ошибка по X координате', fontsize=16)
        axes[0].set_xlabel('Время (с)', fontsize=14)
        axes[0].set_ylabel('Ошибка (м)', fontsize=14)
        axes[0].grid(True)

        sns.lineplot(x=imu_time, y=position_error_smooth[:, 1], ax=axes[1])
        axes[1].set_title('Ошибка по Y координате', fontsize=16)
        axes[1].set_xlabel('Время (с)', fontsize=14)
        axes[1].set_ylabel('Ошибка (м)', fontsize=14)
        axes[1].grid(True)

        sns.lineplot(x=imu_time, y=position_error_smooth[:, 2], ax=axes[2])
        axes[2].set_title('Ошибка по Z координате', fontsize=16)
        axes[2].set_xlabel('Время (с)', fontsize=14)
        axes[2].set_ylabel('Ошибка (м)', fontsize=14)
        axes[2].grid(True)

        plt.tight_layout()
        plt.show()

        # Plot errors for velocities
        fig, axes = plt.subplots(3, 1, figsize=(12, 18))
        sns.lineplot(x=imu_time, y=velocity_error_smooth[:, 0], ax=axes[0])
        axes[0].set_title('Ошибка по X скорости', fontsize=16)
        axes[0].set_xlabel('Время (с)', fontsize=14)
        axes[0].set_ylabel('Ошибка (м/с)', fontsize=14)
        axes[0].grid(True)

        sns.lineplot(x=imu_time, y=velocity_error_smooth[:, 1], ax=axes[1])
        axes[1].set_title('Ошибка по Y скорости', fontsize=16)
        axes[1].set_xlabel('Время (с)', fontsize=14)
        axes[1].set_ylabel('Ошибка (м/с)', fontsize=14)
        axes[1].grid(True)

        sns.lineplot(x=imu_time, y=velocity_error_smooth[:, 2], ax=axes[2])
        axes[2].set_title('Ошибка по Z скорости', fontsize=16)
        axes[2].set_xlabel('Время (с)', fontsize=14)
        axes[2].set_ylabel('Ошибка (м/с)', fontsize=14)
        axes[2].grid(True)

        plt.tight_layout()
        plt.show()

        # Plot errors for orientations
        fig, axes = plt.subplots(3, 1, figsize=(12, 18))
        sns.lineplot(x=imu_time, y=orientation_error_smooth[:, 0], ax=axes[0])
        axes[0].set_title('Ошибка по углу крена (Roll)', fontsize=16)
        axes[0].set_xlabel('Время (с)', fontsize=14)
        axes[0].set_ylabel('Ошибка (градусы)', fontsize=14)
        axes[0].grid(True)

        sns.lineplot(x=imu_time, y=orientation_error_smooth[:, 1], ax=axes[1])
        axes[1].set_title('Ошибка по углу тангажа (Pitch)', fontsize=16)
        axes[1].set_xlabel('Время (с)', fontsize=14)
        axes[1].set_ylabel('Ошибка (градусы)', fontsize=14)
        axes[1].grid(True)

        sns.lineplot(x=imu_time, y=orientation_error_smooth[:, 2], ax=axes[2])
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
        position_error, velocity_error, orientation_error = self.calculate_errors()

        logging.info("Position Error: %s", position_error)
        logging.info("Velocity Error: %s", velocity_error)
        logging.info("Orientation Error: %s", orientation_error)

        self.plot_errors(self.imu_data['time'], position_error, velocity_error, orientation_error, start_time, end_time)


class INSWithoutCorrection(BaseINS):
    def integrate_imu_data(self):
        """
        Интеграция данных IMU без коррекции.
        """
        logging.info('Интеграция данных IMU без коррекции')

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

            # Коррекция угла курса по скорости
            vn, ve, _ = velocities[i]
            heading = np.degrees(np.arctan2(ve, vn)) % 360
            orientation_euler = R.from_quat(orientations[i]).as_euler('xyz', degrees=True)
            orientation_euler[2] = heading
            orientations[i] = R.from_euler('xyz', orientation_euler, degrees=True).as_quat()

            # Пересчет координат
            local_velocity = orientation_new.apply(velocities[i])
            positions[i] = positions[i - 1] + local_velocity * self.dt

        self.state['position'] = positions
        self.state['velocity'] = velocities
        self.state['orientation'] = orientations
        logging.info('Интеграция данных IMU без коррекции завершена')


class INSWithCorrection(BaseINS):
    def __init__(self, imu_file: str, reference_file: str, gnss_file: str, heading_file: str):
        self.gnss_file = gnss_file
        super().__init__(imu_file, reference_file, heading_file)
        self.gnss_data = pd.read_csv(gnss_file)

    def integrate_imu_data(self):
        """
        Интеграция данных IMU с периодической коррекцией данных GNSS.
        """
        logging.info('Интеграция данных IMU с коррекцией')

        GRAVITY = 9.8175  # ускорение свободного падения в м/с²

        omegas = np.radians(self.imu_data[['gyro_x', 'gyro_y', 'gyro_z']].values) * self.dt
        accs = self.imu_data[['accel_x', 'accel_y', 'accel_z']].values

        # Коррекция ускорения по оси Z
        accs[:, 2] += GRAVITY

        positions = np.zeros((self.num_samples, 3))
        velocities = np.zeros((self.num_samples, 3))
        orientations = np.zeros((self.num_samples, 4))
        orientations[0] = self.state['orientation'][0]

        gnss_indices = np.searchsorted(self.imu_data['time'], self.gnss_data['time'])

        for i in range(1, self.num_samples):
            delta_rotation = R.from_rotvec(omegas[i - 1])
            orientation_prev = R.from_quat(orientations[i - 1])
            orientation_new = orientation_prev * delta_rotation
            orientations[i] = orientation_new.as_quat()

            acc_world = orientation_new.apply(accs[i - 1])
            velocities[i] = velocities[i - 1] + acc_world * self.dt

            if i in gnss_indices:
                gnss_index = np.where(gnss_indices == i)[0][0]
                positions[i-1] = np.array(geographic_to_local(
                    self.gnss_data['lat'].iloc[gnss_index], self.gnss_data['lon'].iloc[gnss_index],
                    self.origin_lat, self.origin_lon
                ) + (self.gnss_data['alt'].iloc[gnss_index],))
                velocities[i] = [
                    self.gnss_data['VN'].iloc[gnss_index],
                    self.gnss_data['VE'].iloc[gnss_index],
                    self.gnss_data['VD'].iloc[gnss_index]
                ]

            # Коррекция угла курса по магнитометру
            heading_time = self.imu_data['time'][i]
            heading = np.interp(heading_time, self.heading_data['time'], self.heading_data['heading'])
            orientation_euler = R.from_quat(orientations[i]).as_euler('xyz', degrees=True)
            orientation_euler[2] = heading
            orientations[i] = R.from_euler('xyz', orientation_euler, degrees=True).as_quat()

            # Пересчет координат
            local_velocity = orientation_new.apply(velocities[i])
            positions[i] = positions[i - 1] + local_velocity * self.dt

        self.state['position'] = positions
        self.state['velocity'] = velocities
        self.state['orientation'] = orientations
        logging.info('Интеграция данных IMU с коррекцией завершена')


class INSWithVelocityCorrection(BaseINS):
    def __init__(self, imu_file: str, reference_file: str, heading_file: str):
        super().__init__(imu_file, reference_file, heading_file)

    def integrate_imu_data(self):
        """
        Интеграция данных IMU с коррекцией скорости из эталонной траектории.
        """
        logging.info('Интеграция данных IMU с коррекцией скорости')

        GRAVITY = 9.8175  # ускорение свободного падения в м/с²

        omegas = np.radians(self.imu_data[['gyro_x', 'gyro_y', 'gyro_z']].values) * self.dt
        accs = self.imu_data[['accel_x', 'accel_y', 'accel_z']].values

        # Коррекция ускорения по оси Z
        accs[:, 2] += GRAVITY

        positions = np.zeros((self.num_samples, 3))
        velocities = np.zeros((self.num_samples, 3))
        orientations = np.zeros((self.num_samples, 4))
        orientations[0] = self.state['orientation'][0]

        reference_indices = np.searchsorted(self.imu_data['time'], self.reference_data['time'])

        for i in range(1, self.num_samples):
            delta_rotation = R.from_rotvec(omegas[i - 1])
            orientation_prev = R.from_quat(orientations[i - 1])
            orientation_new = orientation_prev * delta_rotation
            orientations[i] = orientation_new.as_quat()

            acc_world = orientation_new.apply(accs[i - 1])
            velocities[i] = velocities[i - 1] + acc_world * self.dt

            if i in reference_indices:
                ref_index = np.where(reference_indices == i)[0][0]
                velocities[i] = [
                    self.reference_data['VN'].iloc[ref_index],
                    self.reference_data['VE'].iloc[ref_index],
                    self.reference_data['VD'].iloc[ref_index]
                ]

            # Коррекция угла курса по магнитометру
            heading_time = self.imu_data['time'][i]
            heading = np.interp(heading_time, self.heading_data['time'], self.heading_data['heading'])
            orientation_euler = R.from_quat(orientations[i]).as_euler('xyz', degrees=True)
            orientation_euler[2] = heading
            orientations[i] = R.from_euler('xyz', orientation_euler, degrees=True).as_quat()

            # Пересчет координат
            local_velocity = orientation_new.apply(velocities[i])
            positions[i] = positions[i - 1] + local_velocity * self.dt

        self.state['position'] = positions
        self.state['velocity'] = velocities
        self.state['orientation'] = orientations
        logging.info('Интеграция данных IMU с коррекцией скорости завершена')


def main():
    imu_file = 'imu.csv'
    gnss_file = 'gnss.csv'
    reference_file = 'reference_trajectory.csv'
    heading_file = 'heading.csv'
    start_time = 2
    end_time = 1750

    # Запуск с коррекцией по GNSS
    ins_with_correction = INSWithCorrection(imu_file, reference_file, gnss_file, heading_file)
    ins_with_correction.run(start_time, end_time)

    # # Запуск без коррекции
    # ins_without_correction = INSWithoutCorrection(imu_file, reference_file, heading_file)
    # ins_without_correction.run(start_time, end_time)
    #
    # # Запуск с коррекцией по скорости
    # ins_with_velocity_correction = INSWithVelocityCorrection(imu_file, reference_file, heading_file)
    # ins_with_velocity_correction.run(start_time, end_time)


if __name__ == "__main__":
    main()
