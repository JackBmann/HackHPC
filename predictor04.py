"""
predictor01.py
by Ted Morin

read in stock data, review data
fit stock data to review data
(fit from multiple variables)
(fit multiple variables, offload to file)

Usage: python predictorXX.py stockFileName.csv reviewFileName.csv outTag
"""


import numpy as np
import pandas as pd
from sys import argv
#from sklearn import linear_model as lm
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import datetime

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
print stoDataLen
# read in reviews
# expects a csv that has <time, review score> in each row
revData = pd.read_csv(revDataName, sep=',')
revDataWidth = revData.shape[1] 

newKeys = []
for k in revData.keys(): 
    newKeys.append(k+"RollAvg")
newKeys.append("revRollCnt")

indices = []
newData = [[0.0]*stoDataLen]*revDataWidth
for ix, rev in revData.iterrows():
    t = rev[0]
    # get the time related to that date
    for i, stockTime in enumerate(stoData[stoDataKeys[0]]):
        if stockTime > t:
            indices.append((i, t, stockTime))
            break
    i -= 1 
    # update all values within the rolling average
    for j in range(i, min(stoDataLen, i+rollLength)):
        for d in range(revDataWidth-1):
            newData[d][j] += float(rev[d+1])
        newData[revDataWidth-1][j] += 1
    for d in range(revDataWidth-1):
        newData[d][i] = newData[d][i]/newData[revDataWidth-1][i]
newData = np.array(newData).T

print min(indices)

# iterate over stocks and match
fitData = []
fitKeys = []
for k in stoDataKeys[1:]:
    fitKeys.append(k)
    # fit the data
    #fitData.append(lm.LinearRegression(newData, stoData[k].values))
    #print len(stoData[k].values), len(newData[0])
    fitData.append(sm.OLS(stoData[k].values, newData).fit())
    # test how the regression performs
    print fitData[-1].summary()
    prediction = fitData[-1].predict(newData)
    with open(outTag+"_model_"+k+"_"+str(datetime.now())[:10]+".csv",'w') as outf:
        outf.write(k+'\n')
        outf.write(str(newKeys)+'\n')
        outf.write(str(fitData[-1].params))
    with open(outTag+"_predict_"+k+"_"+str(datetime.now())[:10]+".csv",'w') as outf:
        outf.write(k+'\n')
        for pred in list(prediction):
            outf.write(str(pred)+'\n')
    #plt.plot(stoData)
    #plt.plot(stoData[stoDataKeys[0]], prediction)
    #plt.show()

