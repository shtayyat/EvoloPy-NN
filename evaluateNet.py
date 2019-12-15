# -*- coding: utf-8 -*-
"""
Created on Thu May 23 11:59:16 2019

@author: AH
"""

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,r2_score
import numpy as np
import time

def evaluateNet(trainInput,trainOutput,net,isClassifier=True):
    if isClassifier:
        printAcc=[]
    pred=net.sim(trainInput).reshape(len(trainOutput))

    if isClassifier:
        pred=np.round(pred).astype(int)   
        trainOutput=trainOutput.astype(int) 
        pred=np.clip(pred, 0, 1)
        ConfMatrix=confusion_matrix(trainOutput, pred, labels=[0,1])
        ConfMatrix1D=ConfMatrix.flatten()
        time.sleep(5)
        printAcc.append(accuracy_score(trainOutput, pred,normalize=True))
        classification_results= np.concatenate((printAcc,ConfMatrix1D))
        results = classification_results

    else: #For regression
        e = pred - trainOutput
        MSE = ((e) ** 2).mean(axis=None)
        RMSE = np.sqrt(MSE)
        MAE = (abs(e)).mean(axis=None)
        VAF = (1- np.var(e)/np.var(trainOutput))*100
        MMRE = (abs(e)/(trainOutput+0.00001)).mean(axis=None)
        ED = np.sum((e)**2)
        r2 = r2_score(trainOutput, pred)
        time.sleep(5)
        results= np.concatenate(([MSE],[MAE],[RMSE],[MMRE],[VAF],[ED],[r2]))

    return results