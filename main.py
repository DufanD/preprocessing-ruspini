import csv

with open('data/ruspini.csv', 'r') as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
        print(row)

csvFile.close()