import numpy as np
import pandas as pd


# for i in range(2, 7):
#     df = pd.read_csv(f"{i}/im_dist.csv")
#     pogr = pd.read_csv("beacon_coords.csv")["pogr"]
#     for j in range(1, 9):
#         # df[str(j)] -= pogr[j-1]
#         df[str(j)] -= 6
#     df.to_csv(f"{i}/im_dist.csv", index=False)


def zero_iz_nach(df):
    df_filter = df.drop(df[(df < 1).all(axis=1)].index)
    df_filter.reset_index(drop=True, inplace=True)
    df_filter.index += 1
    # Подсчитываем количество ненулевых значений в каждой строке
    non_zero_count = df_filter.astype(bool).sum(axis=1)

    # Фильтруем таблицу, оставляя только строки с менее или равно 3 ненулевыми значениями
    filtered_df = df_filter[non_zero_count > 3]
    return filtered_df


def remove_rows2(df):
    for column in df.columns:
        if column == "time":
            continue
        else:
            mean_val = abs(df[column].mean())
            if np.isnan(mean_val) or column == "h":
                mean_val = 0
            df = df[~(abs(df[column]) > (mean_val + 100))]
            mean_val = abs(df[column].mean())
            df = df[~(abs(df[column]) < (mean_val - 15))]
    df.reset_index(drop=True, inplace=True)
    df.index += 1
    return df


# for i in range(1, 9):
#     df = pd.read_csv(f"{i}/rtsln.csv")
#     df_filter = remove_rows2(df)
#     df_filter.to_csv(f"{i}/rtsln.csv", index=False)

for i in range(1, 9):
    df = pd.read_csv(f"{i}/im_dist.csv")

    print(i)
    # Функция для генерации случайного значения и прибавления его к текущему значению
    def generate_random_value():
        random_value = np.random.uniform(-1, 1) / 100  # Генерация случайного значения от 1 до 70
        return random_value


    # Проход по каждой ячейке и добавление случайного значения
    for column in df.columns:
        if column != 'time':
            for index, value in df[column].items():
                if value >= 5:
                    df.at[index, column] += generate_random_value()

    df.to_csv(f"{i}/im_dist.csv", index=False)
