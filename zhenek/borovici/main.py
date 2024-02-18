import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from zhenek.funcs import dist_least


def rachet(df):
    beacon_coordinates = (
        # (52.1570000000000, 21.9930000000000, 0),
        (138.984000000000, -92.6810000000000, 0),
        (78.1890000000000, -139.061000000000, 0),
        # (-24.4290000000000, -31.2520000000000, 0),
        (-49.9940000000000, 63.2610000000000, 0),
        (20.3570000000000, -96.5530000000000, 0),
    )

    count = 1000
    dist_with_pogr = [0] * count
    for i in range(count):
        dist_with_pogr[i] = [df["2"][i]]
        dist_with_pogr[i].append(df["3"][i])
        dist_with_pogr[i].append(df["6"][i])
        dist_with_pogr[i].append(df["7"][i])

        print(f'{i} / {count}')

    x_dist = [0] * count
    y_dist = [0] * count
    for k in range(count):
        x_dist[k], y_dist[k], _ = dist_least(
            beacon_coordinates, dist_with_pogr[k], 0, 0, 0
        )
        print(f'{k} / {count}')
    plt.style.use("ggplot")
    plt.plot(df['time'][:count], x_dist, label='x')
    plt.plot(df['time'][:count], y_dist, label='y')
    # plt.plot(df['time'], df['4'] - np.mean(df['4']), label='РЭМ-4')
    plt.legend()
    plt.xlabel('Время, с')
    plt.ylabel('Координаты обьекта, м')
    plt.show()




def dist_sko(df):
    plt.style.use("ggplot")
    plt.plot(df['time'], df['2'] - np.mean(df['2']), label='РЭМ-2')
    plt.plot(df['time'], df['3'] - np.mean(df['3']), label='РЭМ-3')
    # plt.plot(df['time'], df['4'] - np.mean(df['4']), label='РЭМ-4')
    plt.legend()
    plt.xlabel('Время, с')
    plt.ylabel('СКО, м')
    plt.show()

    plt.plot(df['time'], df['6'] - np.mean(df['6']), label='РЭМ-6')
    plt.plot(df['time'], df['7'] - np.mean(df['7']), label='РЭМ-7')
    plt.legend()
    # Добавляем подписи к осям
    plt.xlabel('Время, с')
    plt.ylabel('СКО, м')

    # Отображаем график
    plt.show()


def dist(df):
    plt.style.use("ggplot")
    # Строим графики
    plt.plot(df['time'], df['2'], label='РЭМ-2')
    plt.plot(df['time'], df['3'], label='РЭМ-3')
    plt.plot(df['time'], df['4'], label='РЭМ-4')
    plt.plot(df['time'], df['6'], label='РЭМ-6')
    plt.plot(df['time'], df['7'], label='РЭМ-7')

    # Добавляем легенду
    plt.legend()

    # Добавляем подписи к осям
    plt.xlabel('Время, с')
    plt.ylabel('Дистанции, м')

    # Отображаем график
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv("im_dist.csv")
    dist(df)
    dist_sko(df)
    rachet(df)
