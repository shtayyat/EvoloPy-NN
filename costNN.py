# -*- coding: utf-8 -*-
"""
Created on Thu May 23 14:33:56 2019

@author: AH
"""
import numpy as np
import create_NN as cnn


def costNN(x,trainInput,trainOutput,net,HiddenLayersCount,isClassifier=True):
    numInputs = np.shape(trainInput)[1]  # number of inputs
    (net, numNeurons) = cnn.create_NN(x, numInputs, HiddenLayersCount, isClassifier)

    pred = net.sim(trainInput).reshape(len(trainOutput))

    mse = ((pred - trainOutput) ** 2).mean(axis=None)
    neuron_cost = 0  # len(net.layers)/300 + numNeurons/(300*(len(net.layers)-0.98))
    # *(len(net.layers)+0.1)/(len(net.layers)+0.01)) #0.5*((len(net.layers) + numNeurons/(3*numInputs))/HiddenLayersCount)
    return mse + neuron_cost