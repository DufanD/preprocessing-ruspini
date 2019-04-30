#%%Stage1
import impyute as impy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdb
from sklearn.neighbors import KNeighborsClassifier

init_ruspini = pd.read_csv('data/ruspini.csv')
init_ruspini = np.array(init_ruspini, dtype=float)
init_ruspini = pd.DataFrame({
    'x': init_ruspini[:, 0],
    'y': init_ruspini[:, 1],
    'label': init_ruspini[:, 2]
})

ruspini_missing = pd.read_csv('data/data_ruspini_missing.csv')
ruspini_missing = ruspini_missing.replace('?', np.nan)
ruspini_missing = np.array(ruspini_missing, dtype=float)

ruspini_missing = pd.DataFrame({
    'x': ruspini_missing[:, 0],
    'y': ruspini_missing[:, 1],
    'label': ruspini_missing[:, 2]
})

ruspini_missing_grouped = ruspini_missing.groupby('label')

new_ruspini_grouped = list()
for key, item in ruspini_missing_grouped:
    temp = list(impy.fast_knn(np.array(item), k=3))
    for i in temp:
        new_ruspini_grouped.append(i)

new_ruspini_grouped = np.array(new_ruspini_grouped)
#%%
new_ruspini = pd.DataFrame({
    'x': new_ruspini_grouped[:, 0],
    'y': new_ruspini_grouped[:, 1],
    'label': new_ruspini_grouped[:, 2]
})

print(init_ruspini)
print(ruspini_missing)
print(new_ruspini)

plt.figure('Ruspini Origin')
plt.scatter(init_ruspini['x'].values,
            init_ruspini['y'].values,
            c=init_ruspini['label'].values)
plt.xlabel('X')
plt.ylabel('Y')

plt.figure('Ruspini Missing')
plt.scatter(ruspini_missing['x'].values,
            ruspini_missing['y'].values,
            c=ruspini_missing['label'].values)
plt.xlabel('X')
plt.ylabel('Y')

plt.figure('New Ruspini')
plt.scatter(new_ruspini['x'].values,
            new_ruspini['y'].values,
            c=ruspini_missing['label'].values)

plt.xlabel('X')
plt.ylabel('Y')
plt.show()
