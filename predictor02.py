"""
predictor01.py
by Ted Morin

read in stock data, review data
fit stock data to review data
(fit from multiple variables)
(fit multiple variables, offload to file)

Usage: python predictorXX.py stockFileName.csv reviewFileName.csv
"""


import numpy as np
import pandas as pd
import csv
from sys import argv
from sklearn import linear_model as lm

# input/output data
stoDataName = argv[1]
revDataName = argv[2]
outTag = argv[3]

# model parameters
rollLength = 30

# read in stock data
stoData = pd.read_csv(stoDataName, sep=',')
stoDataLen = stoData.shape[0]
stoDataKeys = list(stoData.keys())

# read in reviews
# expects a csv that has <time, review score> in each row
revData = pd.read_csv(revDataName, sep=',')
revDataWidth = revData.shape[1] 

newKeys = []
for k in revData.keys(): 
    newKeys.append(k+"RollAvg")
newKeys.append("revRollCnt")

newData = [[0.0]*stoDataLen]*revDataWidth
for ix, rev in revData.iterrows():
    t = rev[0]
    t = int(t) # convert time to DATE ONLY
    i = t # get the time related to that date
    # update all values within the rolling average
    for j in range(i, min(stoDataLen, i+rollLength)):
        for d in range(revDataWidth-1):
            newData[d][j] += float(rev[d+1])
        newData[revDataWidth-1][j] += 1
    for d in range(revDataWidth-1):
        newData[d][i] = newData[d][i]/newData[revDataWidth-1][i]
#newData = np.array(newData)

# iterate over stocks and match
fitData = []
fitKeys = []
for k in stoDataKeys:
    fitKeys.append(k)
    fitData.append(lm.LinearRegression(newData, stoData[k].values))

print fitKeys
print fitData
