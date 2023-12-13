import pandas as pd

imu = pd.read_csv("data/imu.csv")

imu["gyro_z"] = -imu["gyro_z"]

imu["gyro_x"], imu["gyro_y"] = imu["gyro_y"], imu["gyro_x"]

imu["accel_x"], imu["accel_y"] = imu["accel_y"], imu["accel_x"]

imu.to_csv("data/imu.csv", index=False)


