import csv
import pandas as pd
import numpy as np
import impyute as impy


with open('data/ruspini.csv', 'r') as csvFile:
    reader = csv.reader(csvFile)
    # for row in reader:
    # print(row)

csvFile.close()


# DPPS
ruspini_missing = list()
new_ruspini = list()

with open('data/data_ruspini_missing.csv', 'r') as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
        ruspini_missing.append(row)
csvFile.close()

for i, itemi in enumerate(ruspini_missing):
    for j, itemj in enumerate(ruspini_missing[i]):
        if(itemj == '?'):
            ruspini_missing[i][j] = np.nan
        else:
            ruspini_missing[i][j] = float(itemj)

print(ruspini_missing)
print(impy.mean(np.array(ruspini_missing)))
