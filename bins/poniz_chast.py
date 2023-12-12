import pandas as pd

rsln = pd.read_csv('/home/semyon/PycharmProjects/diplom/bins/data/rtsln.csv')
index = range(10000)
osred = pd.DataFrame(columns=['time', 'lat', 'lon', 'alt'], index=index)


for i in range(10 ** 4 - 1):
    osred['time'][i] = rsln['time'][i * 10 + 5]
    osred['lat'][i] = sum(rsln['lat'][i * 10:(i + 1) * 10]) / 10
    osred['lon'][i] = sum(rsln['lon'][i * 10:(i + 1) * 10]) / 10
    osred['alt'][i] = rsln['alt'][i * 10 + 5]
    print(f'{i} / 10000')

osred.to_csv("data/rtsln_10.csv", index=False)
