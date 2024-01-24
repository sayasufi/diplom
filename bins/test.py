import random

from convert import *

# ref = pd.read_csv('data/rodya/vzlet_sns.csv')
# xy = pd.read_csv('data/rodya/vzlet_ref_xy.csv')
# rtsln_moy = pd.read_csv('data/rtsln.csv')
# rtsln = pd.DataFrame(columns=['time', 'lat', 'lon', 'alt', "VN", "VE", "VD"], index=range(10000))
# rtsln1 = pd.DataFrame(columns=['time', 'lat', 'lon', 'alt', "VN", "VE", "VD"], index=range(10000))

df = pd.read_csv('data/distances.csv')

df_subset = df.iloc[10600:20600]

# df_subset.to_csv('data/perekl/perekl_imu.csv', index=False)




#
# for i in range(17700, 27700):
#     rtsln1["time"][i - 17700] = ref["time"][i]
#     rtsln1["lat"][i - 17700] = ref["lat"][i]
#     rtsln1["lon"][i - 17700] = ref["lon"][i]
#     rtsln1["alt"][i - 17700] = ref["alt"][i]
#     rtsln1["VN"][i - 17700] = ref["VN"][i]
#     rtsln1["VE"][i - 17700] = ref["VE"][i]
#     rtsln1["VD"][i - 17700] = ref["VD"][i]
#     print(i)
#
# ref.to_csv('data/rodya/vzlet_sns_150.csv', index=False)
# rtsln1.to_csv('data/rodya/posadka_sns.csv', index=False)

# xy = pd.read_csv('data/reference_trajectory.csv')
#
plt.plot(df_subset["time"], df_subset["dist1"], label="Distance 1")
plt.plot(df_subset["time"], df_subset["dist2"], label="Distance 2")
plt.plot(df_subset["time"], df_subset["dist3"], label="Distance 3")
plt.plot(df_subset["time"], df_subset["dist4"], label="Distance 4")
plt.plot(df_subset["time"], df_subset["dist5"], label="Distance 5")
plt.plot(df_subset["time"], df_subset["dist6"], label="Distance 6")
plt.xlabel("System time, s")
plt.ylabel("Distance, m")
plt.legend()
plt.grid(True, linewidth=1)
plt.savefig('график.png', dpi=600)
plt.show()
plt.clf()
#
# plt.plot(xy['time'], xy['VE'])
# plt.minorticks_on()
# plt.grid(which='major')
# plt.grid(which='minor', linestyle=':')
# plt.tight_layout()
# plt.savefig('график2.png', dpi=600)
# plt.show()
# plt.clf()
#
# plt.plot(xy['time'], xy['VD'])
# plt.minorticks_on()
# plt.grid(which='major')
# plt.grid(which='minor', linestyle=':')
# plt.tight_layout()
# plt.savefig('график3.png', dpi=600)
# plt.show()
# plt.clf()
