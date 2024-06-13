import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from typing import Tuple, Optional

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Устанавливаем стиль для seaborn
sns.set(style="whitegrid")


def geographic_to_local(lat, lon):
    x = lon * 20037508.34 / 180
    y = np.log(np.tan((90 + lat) * np.pi / 360)) / (np.pi / 180)
    y = y * 20037508.34 / 180
    return x, y


class GNSSComparison:
    def __init__(self, gnss_file: str, reference_file: str):
        self.gnss_file = gnss_file
        self.reference_file = reference_file
        self.gnss_data, self.reference_data = self.read_data()

    def read_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Чтение данных из CSV файлов.

        :return: Кортеж с данными GNSS и эталонными данными.
        """
        logging.info('Чтение данных из CSV файлов')
        gnss_data = pd.read_csv(self.gnss_file)
        reference_data = pd.read_csv(self.reference_file)
        logging.info('Данные успешно считаны')
        return gnss_data, reference_data

    def interpolate_reference_data(self, gnss_time: np.ndarray) -> pd.DataFrame:
        """
        Интерполяция эталонных данных до временных меток GNSS.

        :param gnss_time: Временные метки GNSS.
        :return: Интерполированные эталонные данные.
        """
        logging.info('Интерполяция эталонных данных')
        reference_interp = pd.DataFrame()
        reference_interp['time'] = gnss_time
        reference_interp['lat'] = np.interp(gnss_time, self.reference_data['time'], self.reference_data['lat'])
        reference_interp['lon'] = np.interp(gnss_time, self.reference_data['time'], self.reference_data['lon'])
        reference_interp['alt'] = np.interp(gnss_time, self.reference_data['time'], self.reference_data['alt'])
        reference_interp['VN'] = np.interp(gnss_time, self.reference_data['time'], self.reference_data['VN'])
        reference_interp['VE'] = np.interp(gnss_time, self.reference_data['time'], self.reference_data['VE'])
        reference_interp['VD'] = np.interp(gnss_time, self.reference_data['time'], self.reference_data['VD'])
        logging.info('Интерполяция эталонных данных завершена')
        return reference_interp

    def calculate_errors(self, gnss_data: pd.DataFrame, reference_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Расчет ошибок по позициям и скоростям.

        :param gnss_data: Данные GNSS.
        :param reference_data: Эталонные данные.
        :return: Кортеж с ошибками по позициям и скоростям.
        """
        logging.info('Расчет ошибок')
        reference_positions = np.vstack([
            geographic_to_local(reference_data['lat'].values, reference_data['lon'].values)[0],
            # Convert latitude to meters
            geographic_to_local(reference_data['lat'].values, reference_data['lon'].values)[1],
            # Convert longitude to meters
            reference_data['alt'].values  # Altitude is already in meters
        ]).T
        reference_velocities = reference_data[['VN', 'VE', 'VD']].values

        gnss_positions = np.vstack([
            geographic_to_local(gnss_data['lat'].values, gnss_data['lon'].values)[0],  # Convert latitude to meters
            geographic_to_local(gnss_data['lat'].values, gnss_data['lon'].values)[1],  # Convert longitude to meters
            gnss_data['alt'].values  # Altitude is already in meters
        ]).T
        gnss_velocities = gnss_data[['VN', 'VE', 'VD']].values

        position_error = gnss_positions - reference_positions
        velocity_error = gnss_velocities - reference_velocities

        logging.info('Расчет ошибок завершен')
        return position_error, velocity_error

    def plot_errors(self, gnss_time: np.ndarray, position_error: np.ndarray, velocity_error: np.ndarray,
                    start_time: Optional[float] = None, end_time: Optional[float] = None, window: int = 5):
        """
        Построение графиков ошибок по позициям и скоростям на заданном промежутке времени.

        :param gnss_time: Временные метки GNSS.
        :param position_error: Ошибки по позициям.
        :param velocity_error: Ошибки по скоростям.
        :param start_time: Начало промежутка времени для построения графиков.
        :param end_time: Конец промежутка времени для построения графиков.
        :param window: Размер окна для сглаживания данных в секундах.
        """
        if start_time is not None and end_time is not None:
            mask = (gnss_time >= start_time) & (gnss_time <= end_time)
            gnss_time = gnss_time[mask]
            position_error = position_error[mask]
            velocity_error = velocity_error[mask]

        # Определение размера окна в индексах
        window_size = int(window / (gnss_time[1] - gnss_time[0]))

        # Применение скользящего среднего для ошибок
        position_error_smooth = pd.DataFrame(position_error).rolling(window=window_size, min_periods=1,
                                                                     center=True).mean().values
        velocity_error_smooth = pd.DataFrame(velocity_error).rolling(window=window_size, min_periods=1,
                                                                     center=True).mean().values

        # Plot errors for positions
        fig, axes = plt.subplots(3, 1, figsize=(12, 18))
        sns.lineplot(x=gnss_time, y=position_error_smooth[:, 0], ax=axes[0])
        axes[0].set_title('Ошибка по X координате', fontsize=16)
        axes[0].set_xlabel('Время (с)', fontsize=14)
        axes[0].set_ylabel('Ошибка (м)', fontsize=14)
        axes[0].grid(True)

        sns.lineplot(x=gnss_time, y=position_error_smooth[:, 1], ax=axes[1])
        axes[1].set_title('Ошибка по Y координате', fontsize=16)
        axes[1].set_xlabel('Время (с)', fontsize=14)
        axes[1].set_ylabel('Ошибка (м)', fontsize=14)
        axes[1].grid(True)

        sns.lineplot(x=gnss_time, y=position_error_smooth[:, 2], ax=axes[2])
        axes[2].set_title('Ошибка по Z координате', fontsize=16)
        axes[2].set_xlabel('Время (с)', fontsize=14)
        axes[2].set_ylabel('Ошибка (м)', fontsize=14)
        axes[2].grid(True)

        plt.tight_layout()
        plt.show()

        # Plot errors for velocities
        fig, axes = plt.subplots(3, 1, figsize=(12, 18))
        sns.lineplot(x=gnss_time, y=velocity_error_smooth[:, 0], ax=axes[0])
        axes[0].set_title('Ошибка по X скорости', fontsize=16)
        axes[0].set_xlabel('Время (с)', fontsize=14)
        axes[0].set_ylabel('Ошибка (м/с)', fontsize=14)
        axes[0].grid(True)

        sns.lineplot(x=gnss_time, y=velocity_error_smooth[:, 1], ax=axes[1])
        axes[1].set_title('Ошибка по Y скорости', fontsize=16)
        axes[1].set_xlabel('Время (с)', fontsize=14)
        axes[1].set_ylabel('Ошибка (м/с)', fontsize=14)
        axes[1].grid(True)

        sns.lineplot(x=gnss_time, y=velocity_error_smooth[:, 2], ax=axes[2])
        axes[2].set_title('Ошибка по Z скорости', fontsize=16)
        axes[2].set_xlabel('Время (с)', fontsize=14)
        axes[2].set_ylabel('Ошибка (м/с)', fontsize=14)
        axes[2].grid(True)

        plt.tight_layout()
        plt.show()

    def run(self, start_time: Optional[float] = None, end_time: Optional[float] = None):
        """
        Основной метод для запуска всех вычислений и построения графиков.

        :param start_time: Начало промежутка времени для построения графиков.
        :param end_time: Конец промежутка времени для построения графиков.
        """
        gnss_time = self.gnss_data['time']
        reference_interp = self.interpolate_reference_data(gnss_time)
        position_error, velocity_error = self.calculate_errors(self.gnss_data, reference_interp)

        logging.info("Position Error: %s", position_error)
        logging.info("Velocity Error: %s", velocity_error)

        self.plot_errors(gnss_time, position_error, velocity_error, start_time, end_time)


def main():
    gnss_file = 'gnss.csv'
    reference_file = 'reference_trajectory.csv'

    # Запуск сравнения GNSS с эталонной траекторией
    comparison = GNSSComparison(gnss_file, reference_file)
    start_time = 0
    end_time = 1750
    comparison.run(start_time, end_time)


if __name__ == "__main__":
    main()
