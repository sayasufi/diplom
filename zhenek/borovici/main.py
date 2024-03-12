import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab

from zhenek.funcs import dist_least


def rachet(df, xy, gnss, rtsln, num):
    beacon_coordinates = (
        (52.1570000000000, 21.9930000000000, 0),
        (138.984000000000, -92.6810000000000, 0),
        (78.1890000000000, -139.061000000000, 0),
        (-24.4290000000000, -31.2520000000000, 0),
        (0, 0, 0),
        (-49.9940000000000, 63.2610000000000, 0),
        (20.3570000000000, -96.5530000000000, 0),
        (0, 0, 0),
    )

    count = 10000
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
    temp = None
    for k in range(count):
        dist_with_pogr_in = []
        beacon_in = []
        for i in range(8):
            if dist_with_pogr[k][i] > 0.1:
                dist_with_pogr_in.append(dist_with_pogr[k][i])
                beacon_in.append(beacon_coordinates[i])

        if len(beacon_in) > 3:

            x_dist[k], y_dist[k], _ = dist_least(
                beacon_in, dist_with_pogr_in, 0, 0, 0
            )
        else:
            if temp:
                x_dist[k - 1], y_dist[k - 1] = temp
            else:
                temp = x_dist[k - 1], y_dist[k - 1]
        print(f'{k} / {count}')
    print(x_dist[0], y_dist[0])
    plt.style.use("ggplot")

    e = np.mean(xy["E"])
    n = np.mean(xy["N"])
    h = np.mean(xy["h"])

    plt.plot(df['time'][:count], x_dist - e, label='x')
    plt.plot(df['time'][:count], y_dist - n, label='y')
    # plt.plot(df['time'], df['4'] - np.mean(df['4']), label='РЭМ-4')
    plt.legend()
    plt.xlabel('Время, с')
    plt.ylabel('СКО по координатам, м')
    plt.savefig(f'{num}/png/3.png', dpi=600)
    plt.show()
    # print(f"Ошибка МО по координатам: {math.sqrt((np.mean(x_dist) - e) ** 2 + (np.mean(y_dist) - n) ** 2)} м")


    pylab.subplot(1, 2, 1)
    pylab.hist(x_dist - e, bins=20, histtype='step', label=('X',))
    pylab.legend(loc=2)
    plt.xlabel(f'СКО = {round(np.std(x_dist), 2)}; МО по X = {round(np.mean(x_dist) - e, 2)}')

    pylab.subplot(1, 2, 2)
    pylab.hist(y_dist - n, bins=20, histtype='step', label=('Y',))
    pylab.legend(loc=2)
    plt.xlabel(f'СКО = {round(np.std(y_dist), 2)}; МО по Y = {round(np.mean(y_dist) - n, 2)}')
    plt.suptitle("Алгоритм Python")
    plt.savefig(f'{num}/png/4.png', dpi=600)
    pylab.show()

    pylab.subplot(1, 2, 1)
    pylab.hist(rtsln["E"] - e, bins=20, histtype='step', label=('X',))
    pylab.legend(loc=2)
    plt.xlabel(f'СКО = {round(np.std(rtsln["E"]), 2)}; МО по X = {round(np.mean(rtsln["E"]) - e, 2)}')

    pylab.subplot(1, 2, 2)
    pylab.hist(rtsln["N"] - n, bins=20, histtype='step', label=('Y',))
    pylab.legend(loc=2)
    plt.xlabel(f'СКО = {round(np.std(rtsln["N"]), 2)}; МО по Y = {round(np.mean(rtsln["N"]) - n, 2)}')
    plt.suptitle("Алгоритм С")
    plt.savefig(f'{num}/png/5.png', dpi=600)
    pylab.show()

    pylab.subplot(1, 2, 1)
    pylab.hist(gnss["E"] - e, bins=20, histtype='step', label=('X',))
    pylab.legend(loc=2)
    plt.xlabel(f'СКО = {round(np.std(gnss["E"]), 2)}; МО по X = {round(np.mean(gnss["E"]) - e, 2)}')

    pylab.subplot(1, 2, 2)
    pylab.hist(gnss["N"] - n, bins=20, histtype='step', label=('Y',))
    pylab.legend(loc=2)
    plt.xlabel(f'СКО = {round(np.std(gnss["N"]), 2)}; МО по Y = {round(np.mean(gnss["N"]) - n, 2)}')
    plt.suptitle("Спутник")
    plt.savefig(f'{num}/png/6.png', dpi=600)
    pylab.show()


def dist_sko(df, num):
    plt.style.use("ggplot")
    valid = 0
    for i in range(1, 9):
        if df[str(i)][1000] > 50:
            valid += 1
    fig, axs = plt.subplots(valid, 1, figsize=(8, 10))
    valid = 0
    for i in range(1, 9):
        if df[str(i)][1000] > 50:
            mean = np.mean(df[str(i)])
            a = pd.DataFrame(df[str(i)])
            # a = a.drop(a[a[str(i)] > mean + 50].index)[:10000]
            # print(a[str(i)])
            # print(df['time'][:1000])

            axs[valid].plot(df['time'][:10000], a[str(i)][:10000] - np.mean(a[str(i)][:10000]))
            axs[valid].set_title(f'РЭМ-{i}')
            axs[valid].set_xlabel('Время, с')
            axs[valid].set_ylabel('СКО, м')
            # axs[valid].set_ylim(-0.2, 0.2)
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
        if df[str(i)][1000] > 50:
            mean = np.mean(df[str(i)])
            a = pd.DataFrame(df[str(i)])
            # a = a.drop(a[a[str(i)] > mean + 50].index)[:10000]
            plt.plot(df['time'][:10000], a[:10000], label=f'РЭМ-{i}')

    # Добавляем легенду
    plt.legend()

    # Добавляем подписи к осям
    plt.xlabel('Время, с')
    plt.ylabel('Дистанции, м')

    plt.savefig(f'{num}/png/1.png', dpi=600)
    # Отображаем график
    plt.show()


if __name__ == '__main__':
    for i in (3, 4, 5, 7):
        df = pd.read_csv(f"{i}/im_dist.csv")
        xyz = pd.read_csv(f"{i}/xyz.csv")
        gnss = pd.read_csv(f"{i}/gnss.csv")
        rtsln = pd.read_csv(f"{i}/rtsln.csv")
        dist(df, i)
        dist_sko(df, i)
        rachet(df, xyz, gnss, rtsln, i)
