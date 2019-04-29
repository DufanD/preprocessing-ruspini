#%%Stage1
import impyute as impy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('data/ruspini.csv')

ruspini_missing = pd.read_csv('data/data_ruspini_missing.csv')
ruspini_missing = ruspini_missing.replace('?', np.nan)
ruspini_missing = np.array(ruspini_missing, dtype=float)

new_ruspini = impy.fast_knn(ruspini_missing, k=3)
ruspini_missing = pd.DataFrame({
    'x': ruspini_missing[:, 0],
    'y': ruspini_missing[:, 1],
    'label': ruspini_missing[:, 2]
})

print(ruspini_missing)
print(new_ruspini)

new_ruspini = pd.DataFrame({
    'x': new_ruspini[:, 0],
    'y': new_ruspini[:, 1],
    'label': new_ruspini[:, 2]
})

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
