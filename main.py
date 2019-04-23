import impyute as impy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('data/ruspini.csv')

# Visualization Ruspini Dataset
plt.scatter(df['x'].values, df['y'].values, color='r')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# with open('data/ruspini.csv', 'r') as csvFile:
#     reader = csv.reader(csvFile)
#     for row in reader:
#         print(row)

# csvFile.close()


with open('data/ruspini.csv', 'r') as csvFile:
    reader = csv.reader(csvFile)
    # for row in reader:
    # print(row)

csvFile.close()


# DPPS
ruspini_missing = list()
new_ruspini = list()

ruspini_missing = pd.read_csv('data/data_ruspini_missing.csv')

# with open('data/data_ruspini_missing.csv', 'r') as csvFile:
#     reader = csv.reader(csvFile)
#     for row in reader:
#         ruspini_missing.append(row)
# csvFile.close()

print(ruspini_missing)

# for i, itemi in enumerate(ruspini_missing):
#     for j, itemj in enumerate(ruspini_missing[i]):
#         if(itemj == '?'):
#             ruspini_missing[i][j] = np.nan
#         else:
#             ruspini_missing[i][j] = float(itemj)

# new_ruspini = impy.fast_knn(np.array(ruspini_missing))
# print(ruspini_missing)
# print(new_ruspini)
