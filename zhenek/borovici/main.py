import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab

from zhenek.funcs import dist_least

beacon_coordinates = (
    (52.1570000000000, 21.9930000000000, 0.454),
    (138.984000000000, -92.6810000000000, 0.254),
    (78.1890000000000, -139.061000000000, 0.343),
    (-24.4290000000000, -31.2520000000000, 0.14),
    (0, 0, 0),
    (-49.9940000000000, 63.2610000000000, 0.007),
    (20.3570000000000, -96.5530000000000, 0.093),
    (0, 0, 0),
)


def rachet(df, xy, gnss, rtsln, num):
    beacon_coordinates_rachet = (
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
                beacon_in.append(beacon_coordinates_rachet[i])

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

    # Осреднение rtsln

    meanx = np.mean(x_dist[:200])
    meany = np.mean(y_dist[:200])
    filtx = []
    filty = []
    sumx = 0
    sumy = 0
    for j in range(1, 201):
        sumx += x_dist[j - 1]
        sumy += y_dist[j - 1]
        filtx.append(sumx / j - meanx)
        filty.append(sumy / j - meany)

    plt.plot(df['time'][:200], x_dist[:200] - meanx)
    plt.plot(df['time'][:200], filtx)
    plt.ylim(-0.1, 0.1)
    plt.xlabel('Время, с')
    plt.ylabel('X, м')
    plt.title('Осреднение координаты X (РТСЛН)')
    for spine in plt.gca().spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)
    plt.grid(color='gray', alpha=0.7, linestyle='--')
    plt.gca().set_facecolor('white')
    plt.savefig(f'{num}/png/4.png', dpi=600)
    plt.show()

    plt.plot(df['time'][:200], y_dist[:200] - meany)
    plt.plot(df['time'][:200], filty)
    plt.ylim(-0.1, 0.1)
    plt.xlabel('Время, с')
    plt.ylabel('X, м')
    plt.title('Осреднение координаты Y (РТСЛН)')
    for spine in plt.gca().spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)
    plt.grid(color='gray', alpha=0.7, linestyle='--')
    plt.gca().set_facecolor('white')

    plt.savefig(f'{num}/png/5.png', dpi=600)
    plt.show()

    # Осреднение gnss

    meanN = np.mean(gnss["N"][:10000])
    meanE = np.mean(gnss["E"][:10000])
    filtE = []
    filtN = []
    sumN = 0
    sumE = 0
    for j in range(1, 10001):
        sumN += gnss["N"][j - 1]
        sumE += gnss["E"][j - 1]
        filtN.append(sumN / j - meanN)
        filtE.append(sumE / j - meanE)

    plt.plot(df['time'][:10000], gnss["N"][:10000] - meanN)
    plt.plot(df['time'][:10000], filtN)
    plt.ylim(-1, 1)
    plt.xlabel('Время, с')
    plt.ylabel('N, м')
    plt.title('Осреднение координаты N (ГНСС)')
    for spine in plt.gca().spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)
    plt.grid(color='gray', alpha=0.7, linestyle='--')
    plt.gca().set_facecolor('white')
    plt.savefig(f'{num}/png/6.png', dpi=600)
    plt.show()

    plt.plot(df['time'][:10000], gnss["E"][:10000] - meanE)
    plt.plot(df['time'][:10000], filtE)
    plt.ylim(-1, 1)
    plt.xlabel('Время, с')
    plt.ylabel('E, м')
    plt.title('Осреднение координаты E (ГНСС)')
    for spine in plt.gca().spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)
    plt.grid(color='gray', alpha=0.7, linestyle='--')
    plt.gca().set_facecolor('white')

    plt.savefig(f'{num}/png/7.png', dpi=600)
    plt.show()

    e = np.mean(xy["E"])
    n = np.mean(xy["N"])
    h = np.mean(xy["h"])

    pylab.subplot(1, 2, 1)
    pylab.hist(x_dist - e, bins=100, histtype='step', label=('X',), linewidth=1, fill=True, color='gray', alpha=0.5,
               edgecolor='red')
    pylab.legend(loc=2)
    plt.xlabel(f'СКО = {round(np.std(x_dist), 2)}; МО по X = {round(np.mean(x_dist) - e, 2)}')
    for spine in plt.gca().spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)
    plt.grid(color='gray', alpha=0.7, linestyle='--')
    plt.gca().set_facecolor('white')

    pylab.subplot(1, 2, 2)
    pylab.hist(y_dist - n, bins=100, histtype='step', label=('Y',), linewidth=1, fill=True, color='gray', alpha=0.5,
               edgecolor='red')
    pylab.legend(loc=2)
    plt.xlabel(f'СКО = {round(np.std(y_dist), 2)}; МО по Y = {round(np.mean(y_dist) - n, 2)}')
    plt.suptitle("А2")
    for spine in plt.gca().spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)
    plt.grid(color='gray', alpha=0.7, linestyle='--')
    plt.gca().set_facecolor('white')
    plt.savefig(f'{num}/png/8.png', dpi=600)
    pylab.show()

    pylab.subplot(1, 2, 1)
    pylab.hist(rtsln["E"] - e, bins=100, histtype='step', label=('X',), linewidth=1, fill=True, color='gray', alpha=0.5,
               edgecolor='red')
    pylab.legend(loc=2)
    plt.xlabel(f'СКО = {round(np.std(rtsln["E"]), 2)}; МО по X = {round(np.mean(rtsln["E"]) - e, 2)}')
    for spine in plt.gca().spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)
    plt.grid(color='gray', alpha=0.7, linestyle='--')
    plt.gca().set_facecolor('white')

    pylab.subplot(1, 2, 2)
    pylab.hist(rtsln["N"] - n, bins=100, histtype='step', label=('Y',), linewidth=1, fill=True, color='gray', alpha=0.5,
               edgecolor='red')
    pylab.legend(loc=2)
    plt.xlabel(f'СКО = {round(np.std(rtsln["N"]), 2)}; МО по Y = {round(np.mean(rtsln["N"]) - n, 2)}')
    plt.suptitle("А1")
    for spine in plt.gca().spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)
    plt.grid(color='gray', alpha=0.7, linestyle='--')
    plt.gca().set_facecolor('white')
    plt.savefig(f'{num}/png/9.png', dpi=600)
    pylab.show()

    pylab.subplot(1, 2, 1)
    pylab.hist(gnss["E"] - e, bins=100, histtype='step', label=('X',), linewidth=1, fill=True, color='gray', alpha=0.5,
               edgecolor='red')
    pylab.legend(loc=2)
    plt.xlabel(f'СКО = {round(np.std(gnss["E"]), 2)}; МО по X = {round(np.mean(gnss["E"]) - e, 2)}')
    for spine in plt.gca().spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)
    plt.grid(color='gray', alpha=0.7, linestyle='--')
    plt.gca().set_facecolor('white')

    pylab.subplot(1, 2, 2)
    pylab.hist(gnss["N"] - n, bins=100, histtype='step', label=('Y',), linewidth=1, fill=True, color='gray', alpha=0.5,
               edgecolor='red')
    pylab.legend(loc=2)
    plt.xlabel(f'СКО = {round(np.std(gnss["N"]), 2)}; МО по Y = {round(np.mean(gnss["N"]) - n, 2)}')
    plt.suptitle("Спутник")
    for spine in plt.gca().spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)
    plt.grid(color='gray', alpha=0.7, linestyle='--')
    plt.gca().set_facecolor('white')
    plt.savefig(f'{num}/png/10.png', dpi=600)
    pylab.show()

    def disss(point1, point2):
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    points = [
        (np.mean(e), np.mean(n)),
        (np.mean(x_dist), np.mean(y_dist)),
        (np.mean(gnss["E"]), np.mean(gnss["N"])),
        (np.mean(rtsln["E"]), np.mean(rtsln["N"]))
    ]
    labels = ['Ref', 'А2', 'ГНСС', 'А1']
    colors = ["red", "blue", "green", "purple"]
    plt.figure()

    for i, point in enumerate(points):
        plt.scatter(point[0], point[1], label=labels[i], c=colors[i])
        plt.text(point[0], point[1], labels[i], fontsize=12, ha='right')

    # Создаем окружности
    circle1 = plt.Circle(points[0], disss(points[0], points[1]), color='blue', fill=False, linewidth=1.3)
    circle2 = plt.Circle(points[0], disss(points[0], points[2]), color='green', fill=False, linewidth=1.3)
    circle3 = plt.Circle(points[0], disss(points[0], points[3]), color='purple', fill=False, linewidth=1.3)

    # Добавляем окружности на график
    plt.gca().add_artist(circle1)
    plt.gca().add_artist(circle2)
    plt.gca().add_artist(circle3)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Точки на координатной плоскости')
    plt.xlim(points[0][0] - 3, points[0][0] + 3)
    plt.ylim(points[0][1] - 3, points[0][1] + 3)
    plt.xlabel('X, м')
    plt.ylabel('Y, м')
    plt.gca().set_aspect('equal', adjustable='box')
    for spine in plt.gca().spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)
    plt.grid(color='gray', alpha=0.7, linestyle='--')
    plt.gca().set_facecolor('white')

    plt.text(plt.xlim()[1],
             plt.ylim()[1] - 0.2,
             f'А2 - {round(disss(points[0], points[1]), 2)} м.',
             fontsize=12,
             ha='right',
             va='top',
             color='blue')

    plt.text(plt.xlim()[1],
             plt.ylim()[1] - 0.6,
             f'ГНСС - {round(disss(points[0], points[2]), 2)} м.',
             fontsize=12,
             ha='right',
             va='top',
             color='green')

    plt.text(plt.xlim()[1],
             plt.ylim()[1] - 1,
             f'А1 - {round(disss(points[0], points[3]), 2)} м.',
             fontsize=12,
             ha='right',
             va='top',
             color='purple')

    plt.savefig(f'{num}/png/11.png', dpi=600)
    plt.show()


def dist_sko(df, num, xyz):
    raznica = pd.DataFrame(columns=["name", "1", "2", "3", "4", "5", "6", "7", "8"])
    raz_list = ["None"] * 8
    filt_list = ["None"] * 8
    et_list = ["None"] * 8
    plt.style.use("ggplot")

    valid = 0
    for i in range(1, 9):
        if df[str(i)][1000] > 50:
            valid += 1
    fig, axs = plt.subplots(valid, 1, figsize=(8, 10))
    valid = 0
    for i in range(1, 9):
        if df[str(i)][1000] > 50:
            a = pd.DataFrame(df[str(i)])
            etalon = math.sqrt(
                (beacon_coordinates[i - 1][0] - xyz["E"][0]) ** 2 + (
                        beacon_coordinates[i - 1][1] - xyz["N"][0]) ** 2 + (
                        beacon_coordinates[i - 1][2] - xyz["h"][0]) ** 2)
            filt = []
            etalon = [etalon] * 10000
            sum = 0
            for j in range(1, 10001):
                sum += a[str(i)][:10000][j - 1]
                filt.append(sum / j)
                print(j)
            raz_list[i - 1] = abs(etalon[0] - filt[-1])
            et_list[i - 1] = etalon[0]
            filt_list[i - 1] = filt[-1]

            axs[valid].plot(df['time'][:10000], a[str(i)][:10000])
            axs[valid].plot(df['time'][:10000], filt, color="blue", linewidth=2)
            axs[valid].plot(df['time'][:10000], etalon, color="green", linewidth=2)
            axs[valid].set_title(f'РЭМ-{i}')
            axs[valid].set_xlabel('Время, с')
            axs[valid].set_ylabel('Дальность, м')
            for spine in axs[valid].spines.values():
                spine.set_color('black')
                spine.set_linewidth(1.5)
            axs[valid].set_facecolor('white')
            axs[valid].grid(color='gray', alpha=0.7, linestyle='--')
            valid += 1

    # Установка расстояния между подграфиками
    plt.subplots_adjust(hspace=1)
    slovar = pd.DataFrame([{"name": "Разница",
                            "1": raz_list[0],
                            "2": raz_list[1],
                            "3": raz_list[2],
                            "4": raz_list[3],
                            "5": raz_list[4],
                            "6": raz_list[5],
                            "7": raz_list[6],
                            "8": raz_list[7],
                            }])
    raznica = pd.concat([raznica, slovar], ignore_index=True)
    slovar = pd.DataFrame([{"name": "Эталон",
                            "1": et_list[0],
                            "2": et_list[1],
                            "3": et_list[2],
                            "4": et_list[3],
                            "5": et_list[4],
                            "6": et_list[5],
                            "7": et_list[6],
                            "8": et_list[7],
                            }])
    raznica = pd.concat([raznica, slovar], ignore_index=True)
    slovar = pd.DataFrame([{"name": "Алгоритм",
                            "1": filt_list[0],
                            "2": filt_list[1],
                            "3": filt_list[2],
                            "4": filt_list[3],
                            "5": filt_list[4],
                            "6": filt_list[5],
                            "7": filt_list[6],
                            "8": filt_list[7],
                            }])
    raznica = pd.concat([raznica, slovar], ignore_index=True)
    raznica.to_csv(f"{num}/raznica.csv")
    # Отображаем график
    plt.savefig(f'{num}/png/2.png', dpi=600)
    plt.show()

    for i in range(1, 9):
        if df[str(i)][1000] > 50:
            a = pd.DataFrame(df[str(i)])
            mean = np.mean(a[str(i)][:200])
            filt = []
            sum = 0
            for j in range(1, 201):
                sum += a[str(i)][:200][j - 1]
                filt.append(sum / j - mean)

            plt.plot(df['time'][:200], a[str(i)][:200] - mean)
            plt.plot(df['time'][:200], filt)
            plt.ylim(-0.1, 0.1)
            plt.xlabel('Время, с')
            plt.ylabel('Дальность, м')
            plt.title('Осреднение дальностей')
            for spine in plt.gca().spines.values():
                spine.set_color('black')
                spine.set_linewidth(1.5)
            plt.grid(color='gray', alpha=0.7, linestyle='--')
            plt.gca().set_facecolor('white')
            plt.savefig(f'{num}/png/3.png', dpi=600)
            plt.show()
            break


def dist(df, num, xyz):
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
    plt.title("Дальности")
    for spine in plt.gca().spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)
    plt.gca().set_facecolor('white')
    plt.grid(color='gray', alpha=0.7, linestyle='--')
    plt.savefig(f'{num}/png/1.png', dpi=600)
    # Отображаем график
    plt.show()


if __name__ == '__main__':
    for i in (3, 4, 5, 7):
        df = pd.read_csv(f"{i}/im_dist.csv")
        xyz = pd.read_csv(f"{i}/xyz.csv")
        gnss = pd.read_csv(f"{i}/gnss.csv")
        rtsln = pd.read_csv(f"{i}/rtsln.csv")
        # dist(df, i, xyz)
        dist_sko(df, i, xyz)
        # rachet(df, xyz, gnss, rtsln, i)

    # for i in (1, 2, 6, 8):
    #     df = pd.read_csv(f"{i}/im_dist.csv")
    #     xyz = pd.read_csv(f"{i}/xyz.csv")
    #     # gnss = pd.read_csv(f"{i}/gnss.csv")
    #     # rtsln = pd.read_csv(f"{i}/rtsln.csv")
    #     dist(df, i, xyz)
    #     dist_sko(df, i, xyz)
    #     # rachet(df, xyz, gnss, rtsln, i)
