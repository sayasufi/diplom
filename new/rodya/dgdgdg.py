# import pandas as pd
#
# # Загрузите данные из CSV файлов
# imu_df = pd.read_csv('imu.csv')
# ref_df = pd.read_csv('ref.csv')
#
# # Отфильтруйте строки в ref_df, где хотя бы одно из значений в указанных столбцах равно 0
# filtered_ref_df = ref_df[(ref_df['lat'] != 0) &
#                          (ref_df['lon'] != 0) &
#                          (ref_df['alt'] != 0) &
#                          (ref_df['VE'] != 0) &
#                          (ref_df['VN'] != 0) &
#                          (ref_df['VD'] != 0)]
#
# # Получите индексы строк для сохранения
# indices_to_keep = filtered_ref_df.index
#
# # Отфильтруйте imu_df, используя те же индексы
# filtered_imu_df = imu_df.loc[indices_to_keep]
#
# # Сохраните отфильтрованные DataFrame в новые CSV файлы
# filtered_ref_df.to_csv('ref.csv', index=False)
# filtered_imu_df.to_csv('imu.csv', index=False)

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('trajectory_error.csv')
plt.plot(df['time'], df['pitch'])
plt.show()


