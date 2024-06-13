from scipy.signal import savgol_filter

from convert import *


def geo_to_xy():
    df = pd.read_csv("data/rtsln_filter.csv")
    df_xy = pd.DataFrame()
    df_xy["time"] = df["time"]
    df_xy["x"] = df["lat"]
    df_xy["y"] = df["lon"]
    for i in range(df["lat"].shape[0]):
        df_xy["x"][i], df_xy["y"][i] = geographic_to_local(
            df["lat"][i], df["lon"][i]
        )
        print(i)
    df_xy["alt"] = df["alt"]

    df_xy.to_csv("data/rtsln_filter_xy.csv", index=False)


def calculate_smoothed_velocities(data, frequency):
    # Вычисляем время между соседними измерениями
    time_step = 1 / frequency

    # Вычисляем разницу между соседними значениями
    differences = np.diff(data)

    # Вычисляем скорости, разделив разницу на время
    velocities = [0] * len(data)
    for i in range(len(differences) - 10):
        velocities[i + 5] = sum(differences[i:i + 5]) / 2

    # Применяем фильтр скользящего среднего для сглаживания скоростей
    smoothed_velocities = np.convolve(velocities, np.ones(30) / 30, mode="same")

    smoothed_data = savgol_filter(smoothed_velocities, window_length=30, polyorder=3)

    return smoothed_data


df = pd.read_csv("data/rtsln_filter_xy.csv")
df["VN"] = pd.Series(calculate_smoothed_velocities(list(df["y"]), 50))
df["VE"] = pd.Series(calculate_smoothed_velocities(list(df["x"]), 50))
df["VD"] = pd.Series(calculate_smoothed_velocities(list(df["alt"]), 50))
df.to_csv("data/rtsln_filter_xy.csv", index=False)

# df = pd.read_csv("data/rtsln_filter_xy.csv")
# df1 = pd.read_csv("data/vzlet.csv")
# df1["VN"] = df["VN"]
# df1["VE"] = df["VE"]
# df1["VD"] = df["VD"]
# df1.to_csv("data/vzlet.csv", index=False)

# df = pd.read_csv("data/imu.csv")
# df["accel_z"] = df["accel_z"] * -1
# df.to_csv("data/imu.csv", index=False)
