import csv
import impyute as impy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('data/ruspini.csv')

plt.scatter(df['x'].values, df['y'].values, color='r')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

ruspini_missing = pd.read_csv('data/data_ruspini_missing.csv')
ruspini_missing = ruspini_missing.replace('?', np.nan)

new_ruspini = impy.fast_knn(np.array(ruspini_missing, dtype=float))
print(ruspini_missing)
print(new_ruspini)

# for data in new_ruspini:
#     # x, y = data
#     print(data)

# plt.scatter(x, y, color='g')
# plt.show()
