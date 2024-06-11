import numpy as np
from scipy.spatial.transform import Rotation as R
import logging

class InertialNavigationSystem:
    def __init__(self, imu_data, state, dt):
        self.imu_data = imu_data
        self.state = state
        self.dt = dt
        self.num_samples = len(imu_data)

    def integrate_imu_data(self):
        """
        Интеграция данных IMU без коррекции.
        """
        logging.info('Интеграция данных IMU без коррекции')

        GRAVITY = 9.8175  # ускорение свободного падения в м/с²

        omegas = np.radians(self.imu_data[['gyro_x', 'gyro_y', 'gyro_z']].values) * self.dt
        accs = self.imu_data[['accel_x', 'accel_y', 'accel_z']].values

        # Коррекция ускорения по оси Z (вычитание гравитации)
        accs[:, 2] -= GRAVITY

        positions = self.state['position']
        velocities = self.state['velocity']
        orientations = self.state['orientation']
        orientations[0] = self.state['orientation'][0]

        for i in range(1, self.num_samples):
            delta_rotation = R.from_rotvec(omegas[i - 1])
            orientation_prev = R.from_quat(orientations[i - 1])
            orientation_new = orientation_prev * delta_rotation
            orientations[i] = orientation_new.as_quat()

            acc_world = orientation_new.apply(accs[i - 1])
            velocities[i] = velocities[i - 1] + acc_world * self.dt

            # Коррекция угла курса по скорости (требуется более точная техника, например, фильтр)
            vn, ve, _ = velocities[i]
            heading = np.degrees(np.arctan2(ve, vn)) % 360
            orientation_euler = R.from_quat(orientations[i]).as_euler('xyz', degrees=True)
            orientation_euler[2] = heading
            orientations[i] = R.from_euler('xyz', orientation_euler, degrees=True).as_quat()

            # Пересчет координат
            positions[i] = positions[i - 1] + velocities[i] * self.dt

        self.state['position'] = positions
        self.state['velocity'] = velocities
        self.state['orientation'] = orientations
        logging.info('Интеграция данных IMU без коррекции завершена')

# Пример использования
imu_data = {
    'gyro_x': np.random.randn(1000),
    'gyro_y': np.random.randn(1000),
    'gyro_z': np.random.randn(1000),
    'accel_x': np.random.randn(1000),
    'accel_y': np.random.randn(1000),
    'accel_z': np.random.randn(1000)
}
initial_state = {
    'position': np.zeros((1000, 3)),
    'velocity': np.zeros((1000, 3)),
    'orientation': np.zeros((1000, 4))
}
initial_state['orientation'][0] = [1, 0, 0, 0]  # начальная ориентация (кватернион)
dt = 0.01  # временной шаг

ins = InertialNavigationSystem(imu_data, initial_state, dt)
ins.integrate_imu_data()
