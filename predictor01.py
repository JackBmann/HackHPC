"""
predictor01.py
by Ted Morin

read in stock data, review data
fit stock data to review data (linear)

Usage: python predictorXX.py stockFileName.csv reviewFileName.csv
"""


import numpy as np
import pandas as pd
import csv
from sys import argv

stoDataName = argv[1]
revDataName = argv[2]

# name of stock (column label)
stockName = u' "Spring 2 (m)"'

# read in stock data
stoData = pd.read_csv(stoDataName, sep=',')
stoDataLen = stoData.shape[0]

# read in reviews
# expects a csv that has <time, review score> in each row
revData = pd.read_csv(revDataName, sep=',')

# build rolling average
rollLength = 30
#revList = list(revData)
revRollAvg = [0.0]*stoDataLen
revRollCnt = [0.0]*stoDataLen
for ix, rev in revData.iterrows():
    t = rev[0]
    t = int(t) # convert time to DATE ONLY
    i = t # get the time related to that date
    for j in range(i, min(stoDataLen, i+rollLength)):
        #print rev
        revRollAvg[j] += float(rev[1])
        revRollCnt[j] += 1
    revRollAvg[i] = revRollAvg[i]/revRollCnt[i]
"""
stoData['revRollAvg'] = revRollAvg
stoData['revRollCnt'] = revRollCnt
"""

stoPrices = stoData[u' "Spring 2 (m)"'].values

fit = np.polyfit(revRollAvg, stoPrices, 1)

print fit
