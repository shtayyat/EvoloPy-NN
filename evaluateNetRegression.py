# -*- coding: utf-8 -*-
"""
Created on Thu May 23 11:59:16 2019

@author: AH
"""

from sklearn.metrics import confusion_matrix
#from sklearn.metrics import recall_score
#from sklearn.metrics import precision_score
#from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score,r2_score
import numpy as np
import neurolab as nl
import time
import costNN as cnn

def evaluateNetRegression(solution,trainInput,trainOutput,net,isClassifier=True, MaxHiddenLayers=10):

    
    x=solution.bestIndividual
    
    if isClassifier:
        printAcc=[]

#    numInputs=np.shape(trainInput)[1] #number of inputs
#    numHiddenOutputLayers = len(net.layers)
#    numHiddenLayersX_1 = np.shape(net.layers[0].np['w'][:])[0]
#    
#    split = [(numHiddenLayersX_1*numInputs,numHiddenLayersX_1)]
#    split_sum = [0, split[0][0],split[0][0]+split[0][1]]
#    
#    for i in range(1, numHiddenOutputLayers):
#        numHiddenLayersX = np.shape(net.layers[i].np['w'][:])[0]       
#        split = split+[(numHiddenLayersX*numHiddenLayersX_1,numHiddenLayersX)]
#        split_sum = split_sum + [split_sum[-1] + split[i][0]]
#        split_sum = split_sum + [split_sum[-1] + split[i][1]]       
#        numHiddenLayersX_1 = numHiddenLayersX
#
#    
#    for i in range(numHiddenOutputLayers):
#        net.layers[i].np['w'][:] = x[split_sum[2*i]:split_sum[2*i+1]].reshape(net.layers[i].np['w'][:].shape)
#        net.layers[i].np['b'][:] = x[split_sum[2*i+1]:split_sum[2*i+2]].reshape(net.layers[i].np['b'][:].shape)


    pred=net.sim(trainInput).reshape(len(trainOutput))

    if isClassifier:
        pred=np.round(pred).astype(int)   
        trainOutput=trainOutput.astype(int) 
        pred=np.clip(pred, 0, 1)
		
		#print(ConfMatrix1D)
        ConfMatrix=confusion_matrix(trainOutput, pred)
        ConfMatrix1D=ConfMatrix.flatten()
        time.sleep(5)
        printAcc.append(accuracy_score(trainOutput, pred,normalize=True)) 
        classification_results= np.concatenate((printAcc,ConfMatrix1D))
        results = classification_results

    else:
	#For regression
        e = pred - trainOutput
        MSE = ((e) ** 2).mean(axis=None)
        RMSE = np.sqrt(MSE)
        MAE = (abs(e)).mean(axis=None)
        VAF = (1- np.var(e)/np.var(trainOutput))*100
        MMRE = (abs(e)/(trainOutput+0.00001)).mean(axis=None)
        ED = np.sum((e)**2)
        
        #y_mean = np.mean(pred)
        #r2 = 1 - np.sum(e**2)/np.sum((pred-y_mean)**2)
        r2 = r2_score(trainOutput, pred)
        time.sleep(5)
        results= np.concatenate(([MSE],[MAE],[RMSE],[MMRE],[VAF],[ED],[r2]))

    return results