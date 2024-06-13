import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('trajectory_error.csv')
plt.plot(df['time'], df['pitch'])
plt.show()