import matplotlib.pyplot as plt
import pandas as pd


def change(file):
    # Загрузка данных из csv файла
    df = pd.read_csv(file)
    # Создание нового столбца 'Изменение высоты'
    df['time'] = df['time']

    # Сохранение измененных данных в csv файл
    df.to_csv('reference_trajectory.csv', index=False)


def delete(file):
    # Загрузить файл CSV в DataFrame
    df = pd.read_csv(file)

    # Удалить n строк из начала таблицы
    n = 5783
    df = df.drop(range(n))

    # Сохранить изменения в файле CSV
    df.to_csv(file, index=False)


# Загрузка данных из файла CSV
data = pd.read_csv('reference_trajectory.csv')

# Извлечение значений двух столбцов
x = data['lat']
y = data['lon']

# Построение графика
plt.plot(y, x)
plt.xlabel('Название оси X')
plt.ylabel('Название оси Y')
plt.title('Название графика')
plt.show()
