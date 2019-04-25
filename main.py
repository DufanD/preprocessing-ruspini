import csv
import impyute as impy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('data/ruspini.csv')

ruspini_missing = pd.read_csv('data/data_ruspini_missing.csv')
ruspini_missing = ruspini_missing.replace('?', np.nan)

new_ruspini = impy.fast_knn(np.array(ruspini_missing, dtype=float))
print(ruspini_missing)
print(new_ruspini)

new_ruspini = pd.DataFrame(
    {'x': new_ruspini[:, 0], 'y': new_ruspini[:, 1], 'label': new_ruspini[:, 2]})

plt.figure(1)
plt.subplot(111)
plt.scatter(df['x'].values, df['y'].values,
            c=df['label'].values, label='ruspini missing')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(loc='upper left')

plt.figure(2)
plt.scatter(new_ruspini['x'].values,
            new_ruspini['y'].values, c=df['label'].values+1, label='new ruspini')

plt.xlabel('X')
plt.ylabel('Y')
plt.legend(loc='upper left')
plt.show()
