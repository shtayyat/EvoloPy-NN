# -*- coding: utf-8 -*-
"""
Created on Thu May 23 14:33:56 2019

@author: AH
"""
import numpy
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

import create_NN as cnn

def costNN(x,trainInput,trainOutput,HiddenLayersCount,isClassifier=True, costFunction = "MSE"):
    numInputs = np.shape(trainInput)[1]  # number of inputs
    (net, numNeurons) = cnn.create_NN(x, numInputs, HiddenLayersCount, isClassifier)

    pred = net.sim(trainInput).reshape(len(trainOutput))
    pred = np.round(pred).astype(int)
    trainOutput = trainOutput.astype(int)
    pred = np.clip(pred, 0, 1)

    if (costFunction == "MSE"):
        mse = ((pred - trainOutput) ** 2).mean(axis=None)
        return mse

    if (costFunction == "Accuracy"):
        acc = accuracy_score(trainOutput, pred, normalize=True)
        return acc

    if (costFunction == "Gmean"):
        confMatrix = confusion_matrix(trainOutput, pred, labels=[0,1]).flatten()
        TP = confMatrix[0]
        FP = confMatrix[1]
        FN = confMatrix[2]
        gMean = TP/ numpy.sqrt((TP+FP)*(TP+FN))

        return gMean
