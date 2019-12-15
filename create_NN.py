# -*- coding: utf-8 -*-
"""
Created on Thu May 23 14:44:59 2019

@author: AH
"""
import numpy as np
import neurolab as nl
#import solution

def create_NN(x,numInputs,HiddenLayersCount, isClassifier=True):
    #numInputs=6
    #HiddenLayersCount=20
    maxNeurons = numInputs*3
    out = 1

    NN_Layers = x[:HiddenLayersCount].copy()
    
    # clip if greater than 1
    NN_Layers[NN_Layers>1]=1
    x[:HiddenLayersCount]= NN_Layers.copy()
        
    NN_Layers = (NN_Layers*maxNeurons).round().astype('int')#denormalize
    
    NN_Layers2 = NN_Layers.copy()
    NN_Layers2[NN_Layers2<2] = 0
    NN_Layers2 = np.append(NN_Layers2,[out])

    NN_Layers = NN_Layers[NN_Layers>=2]#delet layers with less than 2 neurons
    NN_Layers = np.append(NN_Layers,[out])
    
    
    numNeurons = sum(NN_Layers)
    numHiddenOutputLayers = len(NN_Layers)
    numOfHiddenLayers = numHiddenOutputLayers-1
   
    NN_Layers0 = [maxNeurons]*HiddenLayersCount+[1]
    numHiddenOutputLayers0 = len(NN_Layers0)    
    

    net = nl.net.newff([[0, 1]]*numInputs, NN_Layers, transf=[nl.trans.TanSig()]*numOfHiddenLayers+[nl.trans.PureLin()])


    x2= x[HiddenLayersCount:].copy()
    
    split = [(NN_Layers[0]*numInputs,NN_Layers[0])]
    split_sum = [0, split[0][0],split[0][0]+split[0][1]]
    
    for i in range(1, numHiddenOutputLayers):
        split = split+[(NN_Layers[i]*NN_Layers[i-1],NN_Layers[i])]
        split_sum = split_sum + [split_sum[-1] + split[i][0]]
        split_sum = split_sum + [split_sum[-1] + split[i][1]]

    split0 = [(NN_Layers0[0]*numInputs,NN_Layers0[0])]
    split_sum0 = [0, split0[0][0],split0[0][0]+split0[0][1]]
    
    for i in range(1, numHiddenOutputLayers0):
        split0 = split0+[(NN_Layers0[i]*NN_Layers0[i-1],NN_Layers0[i])]
        split_sum0 = split_sum0 + [split_sum0[-1] + split0[i][0]]
        split_sum0 = split_sum0 + [split_sum0[-1] + split0[i][1]]        

    k=0
    for i in range(HiddenLayersCount+1):
        if (NN_Layers2[i] != 0):
            #print(split_sum0)
            #print(split)
           # print(i)
            input_w = x2[split_sum0[2*i]:split_sum0[2*i]+split[k][0]].reshape(net.layers[k].np['w'][:].shape)
            net.layers[k].np['w'][:] = input_w
            input_b = x2[split_sum0[2*i+1]:split_sum0[2*i+1]+split[k][1]].reshape(net.layers[k].np['b'][:].shape)
            net.layers[k].np['b'][:] = input_b
            k +=1 
        
    return (net, numNeurons)