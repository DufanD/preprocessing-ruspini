import csv
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