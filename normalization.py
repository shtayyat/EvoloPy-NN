import math

import numpy
def minMax(dataset): # weitendorf linear standardization
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

def vectorStandardization(dataset):
    squarRoots = list()
    for i in range(len(dataset[0])):
        col_values = [pow(row[i], 2) for row in dataset]
        column_sum = sum(col_values)
        column_suqar_root = round(math.sqrt(column_sum), 2)
        squarRoots.append(column_suqar_root)
    for row in dataset:
        for i in range(len(row)):
            row[i] = row[i] / squarRoots[i]

def manhattanStandardization(dataset):
    absoluteSums = list()
    for i in range(len(dataset[0])):
        col_values = [abs(row[i]) for row in dataset]
        column_sum_absolutes = sum(col_values)
        absoluteSums.append(column_sum_absolutes)
    for row in dataset:
        for i in range(len(row)):
            row[i] = row[i] / absoluteSums[i]

def maxLinearStandardization(dataset):
    maxValues = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_max = max(col_values)
        maxValues.append(value_max)
    for row in dataset:
        for i in range(len(row)):
            row[i] = row[i] / maxValues[i]

def peldschusNonLinearStandardization(dataset):
    maxValues = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_max = max(col_values)
        maxValues.append(value_max)
    for row in dataset:
        for i in range(len(row)):
            row[i] = pow(row[i] / maxValues[i], 2)

def zafdaksLogarithmicStandardization(dataset):
    lenSums = list()
    for i in range(len(dataset[0])):
        col_values = [numpy.log(row[i]) if row[i] != 0 else 0 for row in dataset]
        column_sum_absolutes = sum(col_values)
        lenSums.append(column_sum_absolutes)
    for row in dataset:
        for i in range(len(row)):
            row[i] = numpy.log(row[i]) / lenSums[i] if lenSums[i] != 0 and row[i] != 0 else 0
