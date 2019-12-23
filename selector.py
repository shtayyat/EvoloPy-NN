# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 20:44:52 2016

@author: hossam
"""
import PSO as pso
import MVO as mvo
import GWO as gwo
import MFO as mfo
import CS as cs
import BAT as bat
import numpy
import costNN
import evaluateNet as evalNet  # updated
import create_NN as cnn
import normalization
def selector(algo, func_details, popSize, Iter, trainDataset, testDataset, isClassifier, HiddenLayersCount, normalizationFunction, costFunction):
    function_name = func_details[0]
    lb = func_details[1]
    ub = func_details[2]
    dataTrain = "datasets/" + trainDataset
    dataTest = "datasets/" + testDataset

    Dataset_train = numpy.loadtxt(open(dataTrain, "rb"), delimiter=",", skiprows=0)
    Dataset_test = numpy.loadtxt(open(dataTest, "rb"), delimiter=",", skiprows=0)
    getattr(normalization,normalizationFunction)(Dataset_train)
    getattr(normalization,normalizationFunction)(Dataset_test)
    numInputsTrain = numpy.shape(Dataset_train)[1] - 1  # number of features in the train dataset

    trainInput = Dataset_train[:, 0:-1]
    trainOutput = Dataset_train[:, -1]
    testInput = Dataset_test[:, 0:-1]
    testOutput = Dataset_test[:, -1]
    if (isClassifier == False):
        # normalization for output
        minY = min(min(trainOutput), min(testOutput))
        maxY = max(max(trainOutput), max(testOutput))
        normalizationFactor = (maxY - minY)

    numberOfHiddenNeurons = numInputsTrain * 3

    # number of hidden neurons
    HiddenNeurons = [numberOfHiddenNeurons] * HiddenLayersCount
    NN_Layers = HiddenNeurons.copy()
    NN_Layers = NN_Layers + [1]

    dim = HiddenLayersCount + (numInputsTrain + 1) * HiddenNeurons[0]

    for i in range(1, HiddenLayersCount + 1):
        dim = dim + (NN_Layers[i - 1] + 1) * NN_Layers[i]

    if (algo == 0):
        x = pso.PSO(getattr(costNN, function_name), lb, ub, dim, popSize, Iter, trainInput,
                    trainOutput, HiddenLayersCount, isClassifier, costFunction)
    if (algo == 1):
        x = mvo.MVO(getattr(costNN, function_name), lb, ub, dim, popSize, Iter, trainInput,
                    trainOutput, HiddenLayersCount, isClassifier, costFunction)
    if (algo == 2):
        x = gwo.GWO(getattr(costNN, function_name), lb, ub, dim, popSize, Iter, trainInput,
                    trainOutput, HiddenLayersCount, isClassifier, costFunction)
    if (algo == 3):
        x = mfo.MFO(getattr(costNN, function_name), lb, ub, dim, popSize, Iter, trainInput,
                    trainOutput, HiddenLayersCount, isClassifier, costFunction)
    if (algo == 4):
        x = cs.CS(getattr(costNN, function_name), lb, ub, dim, popSize, Iter, trainInput,
                  trainOutput, HiddenLayersCount, isClassifier, costFunction)
    if (algo == 5):
        x = bat.BAT(getattr(costNN, function_name), lb, ub, dim, popSize, Iter, trainInput,
                    trainOutput, HiddenLayersCount, isClassifier, costFunction)

    Xsolution = x.bestIndividual
    (net, numNeurons) = cnn.create_NN(Xsolution, numInputsTrain, HiddenLayersCount)
    print("layers count: ", len(net.layers))

    # Evaluate MLP classification model based on the training set
    train_results = evalNet.evaluateNet(trainInput, trainOutput, net, isClassifier)

    # Evaluate MLP classification model based on the testing set
    test_results = evalNet.evaluateNet(testInput, testOutput, net, isClassifier)

    # Final archeticture of network
    Xlayers = Xsolution[:HiddenLayersCount].copy() * numberOfHiddenNeurons
    Xlayers = Xsolution[:HiddenLayersCount].copy()
    Xlayers = (Xlayers * numberOfHiddenNeurons).round().astype('int')  # denormalize
    Xlayers[Xlayers < 2] = 0
    x.layers = Xlayers

    if isClassifier:
        # trainClassification_results
        x.trainAcc = train_results[0]
        x.trainTP = train_results[1]
        x.trainFN = train_results[2]
        x.trainFP = train_results[3]
        x.trainTN = train_results[4]
        print("Train Results:", train_results)
        # testClassification_results
        x.testAcc = test_results[0]
        x.testTP = test_results[1]
        x.testFN = test_results[2]
        x.testFP = test_results[3]
        x.testTN = test_results[4]
        print("Test Results:", test_results)

        # Not applicable results
        x.trainMSE = x.trainMAE = x.testMSE = x.testMAE = x.valMSE = x.valMAE = None
        x.trainRMSE = x.trainMMRE = x.testRMSE = x.testMMRE = x.valRMSE = x.valMMRE = None
        x.trainR2 = x.valR2 = x.testR2 = None
        x.trainVAF = x.trainED = x.testVAF = x.testED = x.valVAF = x.valED = None
    else:
        # trainRegression_results
        x.trainMSE = train_results[0] * (normalizationFactor ** 2)
        x.trainMAE = train_results[1] * normalizationFactor
        x.trainRMSE = train_results[2] * normalizationFactor
        x.trainMMRE = train_results[3]
        x.trainVAF = train_results[4]
        x.trainED = train_results[5] * (normalizationFactor ** 2)
        x.trainR2 = train_results[6]
        print("Training MSE =", x.trainMSE, "|| Training MAE=", x.trainMAE)

        # testRegression_results
        x.testMSE = test_results[0] * (normalizationFactor ** 2)
        x.testMAE = test_results[1] * normalizationFactor
        x.testRMSE = test_results[2] * normalizationFactor
        x.testMMRE = test_results[3]
        x.testVAF = test_results[4]
        x.testED = test_results[5] * (normalizationFactor ** 2)
        x.testR2 = test_results[6]
        print("Testing MSE =", x.testMSE, "|| Testing  MAE=", x.testMAE)

        # Not applicable results
        x.trainAcc = x.trainTP = x.trainFN = x.trainFP = x.trainTN = None
        x.valAcc = x.valTP = x.valFN = x.valFP = x.valTN = None
        x.testAcc = x.testTP = x.testFN = x.testFP = x.testTN = None

    return x

#####################################################################    
