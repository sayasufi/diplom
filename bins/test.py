import pandas as pd
from matplotlib import pyplot as plt

w1 = 1
w2 = 2



# ref = pd.read_csv('data/reference_trajectory.csv')
# rtsln = pd.read_csv('data/rtsln_filter.csv')
#
# for i in range(len(ref)):
#     rtsln["lat"][i] = (ref["lat"][i] * w1 + rtsln["lat"][i] * w2) / (w1 + w2)
#     rtsln["lon"][i] = (ref["lon"][i] * w1 + rtsln["lon"][i] * w2) / (w1 + w2)
#     rtsln["alt"][i] = (ref["alt"][i] * w1 + rtsln["alt"][i] * w2) / (w1 + w2)
#     rtsln["VN"][i] = (ref["VN"][i] * w2 + rtsln["VN"][i] * w1) / (w1 + w2)
#     rtsln["VE"][i] = (ref["VE"][i] * w2 + rtsln["VE"][i] * w1) / (w1 + w2)
#     rtsln["VD"][i] = (ref["VD"][i] * w2 + rtsln["VD"][i] * w1) / (w1 + w2)
#     print(i)
#
# rtsln.to_csv('data/rtsln_filter_ref.csv', index=False)



xy = pd.read_csv('data/reference_trajectory.csv')

plt.plot(xy['time'], xy['VD'])
plt.xlabel("System time, s")
plt.legend()
plt.grid(True, linewidth=1)
plt.savefig('график.png', dpi=600)
plt.show()
plt.clf()

plt.plot(xy['time'], xy['VE'])
plt.minorticks_on()
plt.grid(which='major')
plt.grid(which='minor', linestyle=':')
plt.tight_layout()
plt.savefig('график2.png', dpi=600)
plt.show()
plt.clf()

plt.plot(xy['time'], xy['VD'])
plt.minorticks_on()
plt.grid(which='major')
plt.grid(which='minor', linestyle=':')
plt.tight_layout()
plt.savefig('график3.png', dpi=600)
plt.show()
plt.clf()