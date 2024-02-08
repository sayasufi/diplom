import pandas as pd
from scipy.signal import savgol_filter


def smooth_and_downsample(data):
    # Сглаживание данных с использованием Savitzky-Golay фильтра
    smoothed_data = savgol_filter(data, window_length=100, polyorder=3)

    # Понижение частоты в 10 раз
    # downsampled_data = smoothed_data[::10]

    return smoothed_data


rsln = pd.read_csv('/home/semyon/PycharmProjects/diplom/bins/data/rtsln.csv')
index = range(100000)
osred = pd.DataFrame(columns=['time', 'lat', 'lon', 'alt'], index=index)

# for i in range(10 ** 4):
#     osred['time'][i] = rsln['time'][i * 10 + 5]
#     osred['alt'][i] = rsln['alt'][i * 10 + 5]

osred['time'] = rsln['time']
osred['alt'] = rsln['alt']
osred['lat'] = pd.Series(smooth_and_downsample(list(rsln["lat"])))
osred['lon'] = pd.Series(smooth_and_downsample(list(rsln["lon"])))

osred.to_csv("data/vzlet.csv", index=False)
