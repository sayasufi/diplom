import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab

from zhenek.funcs import dist_least


def rachet(df, xy, num):
    beacon_coordinates = (
        (52.1570000000000, 21.9930000000000, 0),
        (138.984000000000, -92.6810000000000, 0),
        (78.1890000000000, -139.061000000000, 0),
        (-24.4290000000000, -31.2520000000000, 0),
        (-49.9940000000000, 63.2610000000000, 0),
        (0, 0, 0),
        (20.3570000000000, -96.5530000000000, 0),
        (0, 0, 0),
    )

    count = 1000
    dist_with_pogr = [0] * count
    for i in range(count):
        dist_with_pogr[i] = [df["1"][i]]
        dist_with_pogr[i].append(df["2"][i])
        dist_with_pogr[i].append(df["3"][i])
        dist_with_pogr[i].append(df["4"][i])
        dist_with_pogr[i].append(df["5"][i])
        dist_with_pogr[i].append(df["6"][i])
        dist_with_pogr[i].append(df["7"][i])
        dist_with_pogr[i].append(df["8"][i])

        print(f'{i} / {count}')

    x_dist = [0] * count
    y_dist = [0] * count
    for k in range(count):
        dist_with_pogr_in = []
        beacon_in = []
        for i in range(8):
            if dist_with_pogr[k][i] > 0.1:
                dist_with_pogr_in.append(dist_with_pogr[k][i])
                beacon_in.append(beacon_coordinates[i])

        x_dist[k], y_dist[k], _ = dist_least(
            beacon_in, dist_with_pogr_in, 0, 0, 0
        )
        print(f'{k} / {count}')
    plt.style.use("ggplot")
    plt.plot(df['time'][:count], x_dist, label='x')
    plt.plot(df['time'][:count], y_dist, label='y')
    # plt.plot(df['time'], df['4'] - np.mean(df['4']), label='РЭМ-4')
    plt.legend()
    plt.xlabel('Время, с')
    plt.ylabel('Координаты обьекта, м')
    plt.savefig(f'{num}/png/3.png', dpi=600)
    plt.show()

    e = np.mean(xy["E"])
    n = np.mean(xy["N"])
    h = np.mean(xy["h"])
    VE = np.mean(xy["VE"])
    VN = np.mean(xy["VN"])

    plt.plot(df['time'][:count], x_dist - np.mean(x_dist), label='x')
    plt.plot(df['time'][:count], y_dist - np.mean(y_dist), label='y')
    # plt.plot(df['time'], df['4'] - np.mean(df['4']), label='РЭМ-4')
    plt.legend()
    plt.xlabel('Время, с')
    plt.ylabel('СКО по координатам, м')
    plt.savefig(f'{num}/png/4.png', dpi=600)
    plt.show()
    print(f"Ошибка МО по координатам: {math.sqrt((np.mean(x_dist) - e) ** 2 + (np.mean(y_dist) - n) ** 2)} м")

    vx = [0] * count
    vy = [0] * count
    for i in range(count - 1):
        vx[i] = x_dist[i + 1] - x_dist[i]
        vy[i] = y_dist[i + 1] - y_dist[i]

    plt.plot(df['time'][:count], vx - np.mean(vx), label='x')
    plt.plot(df['time'][:count], vy - np.mean(vy), label='y')
    # plt.plot(df['time'], df['4'] - np.mean(df['4']), label='РЭМ-4')
    plt.legend()
    plt.xlabel('Время, с')
    plt.ylabel('СКО по скоростям, м/с')
    plt.savefig(f'{num}/png/5.png', dpi=600)
    plt.show()
    print(f"Ошибка МО по скоростям: {math.sqrt((np.mean(vx) - VE) ** 2 + (np.mean(vy) - VN) ** 2)} м/с")

    pylab.subplot(1, 2, 1)
    pylab.hist(x_dist - e, bins=20, histtype='step', label=('X',))
    pylab.legend(loc=2)
    plt.xlabel(f'СКО = {round(np.std(x_dist), 2)}; МО по X = {round(np.mean(x_dist) - e, 2)}')

    pylab.subplot(1, 2, 2)
    pylab.hist(y_dist - n, bins=20, histtype='step', label=('Y',))
    pylab.legend(loc=2)
    plt.xlabel(f'СКО = {round(np.std(y_dist), 2)}; МО по Y = {round(np.mean(y_dist) - n, 2)}')
    plt.savefig(f'{num}/png/6.png', dpi=600)
    pylab.show()


def dist_sko(df, num):
    plt.style.use("ggplot")
    valid = 0
    for i in range(1, 9):
        if df[str(i)][1000] > 0.1:
            valid += 1
    fig, axs = plt.subplots(valid, 1, figsize=(8, 10))
    valid = 0
    for i in range(1, 9):
        if df[str(i)][1000] > 0.1:
            mean = np.mean(df[str(i)])
            a = pd.DataFrame(df[str(i)][10000:11000])
            a = a.drop(a[a[str(i)] > mean + 50].index)
            # print(a[str(i)])
            # print(df['time'][:1000])


            axs[valid].plot(df['time'][:1000], a[str(i)] - np.mean(a[str(i)]))
            axs[valid].set_title(f'РЭМ-{i}')
            axs[valid].set_xlabel('Время, с')
            axs[valid].set_ylabel('СКО, м')
            valid += 1


    # Установка расстояния между подграфиками
    plt.subplots_adjust(hspace=1)

    # Отображаем график
    plt.savefig(f'{num}/png/2.png', dpi=600)
    plt.show()


def dist(df, num):
    plt.style.use("ggplot")
    # Строим графики
    for i in range(1, 9):
        if df[str(i)][1000] > 0.1:
            mean = np.mean(df[str(i)])
            a = pd.DataFrame(df[str(i)][10000:11000])
            a = a.drop(a[a[str(i)] > mean + 50].index)
            plt.plot(df['time'][:1000], a, label=f'РЭМ-{i}')

    # Добавляем легенду
    plt.legend()

    # Добавляем подписи к осям
    plt.xlabel('Время, с')
    plt.ylabel('Дистанции, м')

    plt.savefig(f'{num}/png/1.png', dpi=600)
    # Отображаем график
    plt.show()


if __name__ == '__main__':
    for i in range(7, 8):
        df = pd.read_csv(f"{i}/im_dist.csv")
        xy = pd.read_csv(f"{i}/xyz_v.csv")
        dist(df, i)
        dist_sko(df, i)
        rachet(df, xy, i)
