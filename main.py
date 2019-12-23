import csv
import numpy
import time
import selector as slctr

'''
Selecting the optimization algorithims to be used in run time
set the algorithim you want to run to True else set them to False
'''
PSO = True
MVO = True
GWO = True
MFO = True
CS = True
BAT = True

optimizer = [PSO, MVO, GWO, MFO, CS, BAT]

'''
Selecting the datasets to be used in run time
set the database in the array of classifier or regression
Make sure you are splitting the datasets with Train and Train
Default Available: ["BreastCancer", "Diabetes", "Liver", "Parkinsons", "Vertebral"]
'''
datasetsClassifier = ["BreastCancer", "Diabetes", "Liver", "Parkinsons", "Vertebral"]
regression_datasets = ["Diabetes"]

'''
choose what you want to use in normalization
available options are: ["minMax","vectorStandardization","manhattanStandardization","maxLinearStandardization","peldschusNonLinearStandardization","zafdaksLogarithmicStandardization"]
'''
normalizationFunction = "minMax"

'''
choose if you want to run a classification or regression
isClassifier = True Then it Will use regression
'''
isClassifier = True

'''
Select number of repetitions for each experiment. 
To obtain meaningful statistical results, usually 30 independent runs 
are executed for each algorithm.
'''
NumOfRuns = 1

# Select general parameters for all optimizers (population size, number of iterations)
PopulationSize = 20
Iterations = 40

'''
Select cost function
Available options are: ["MSE", "Accuracy", "Gmean"]
'''
costFunction = "MSE"

# Export results ?
Export = True

'''
ExportToFile="YourResultsAreHere.csv"
Automaticly generated file name by date and time
'''
ExportToFile = "experiment" + time.strftime("%Y-%m-%d-%H-%M-%S") + ".csv"

# Check if it works at least once
Flag = False

#Number of hidden layers
HiddenLayersCount = 1

# CSV Header for for the cinvergence
CnvgHeader = []

datasets = datasetsClassifier if isClassifier == True else regression_datasets
for l in range(0, Iterations):
    CnvgHeader.append("Iter" + str(l + 1))

trainDataset = "breastTrain.csv"
testDataset = "breastTest.csv"
for j in range(0, len(datasets)):  # specfiy the number of the datasets
    for i in range(0, len(optimizer)):

        if ((optimizer[
                 i] == True)):  # start experiment if an optimizer and an objective function is selected
            for k in range(0, NumOfRuns):

                func_details = ["costNN", -1, 1]
                trainDataset = datasets[j] + "Train.csv"
                testDataset = datasets[j] + "Test.csv"
                x = slctr.selector(i, func_details, PopulationSize, Iterations, trainDataset,
                                   testDataset, isClassifier, HiddenLayersCount, normalizationFunction, costFunction)

                if (Export == True):
                    with open(ExportToFile, 'a', newline='\n') as out:
                        writer = csv.writer(out, delimiter=',')
                        if (Flag == False):  # just one time to write the header of the CSV file
                            if isClassifier:
                                header = numpy.concatenate([["Optimizer", "Dataset", "objfname",
                                                             "Experiment", "startTime", "EndTime",
                                                             "ExecutionTime", "trainAcc", "trainTP",
                                                             "trainFN", "trainFP", "trainTN",
                                                             "testAcc", "testTP", "testFN",
                                                             "testFP", "testTN"], CnvgHeader])
                            else:
                                header = numpy.concatenate([["Optimizer", "Dataset", "objfname",
                                                             "Experiment", "startTime", "EndTime",
                                                             "ExecutionTime", "trainAcc", "trainTP",
                                                             "trainFN", "trainFP", "trainTN",
                                                             "testAcc", "testTP", "testFN",
                                                             "testFP", "testTN", "trainMSE",
                                                             "trainMAE", "trainRMSE", "trainMMRE",
                                                             "trainVAF", "trainED", "trainR2",
                                                             "testMSE", "testMAE", "testRMSE",
                                                             "testMMRE", "testVAF", "testED",
                                                             "testR2", "Archeticture"], CnvgHeader])
                            writer.writerow(header)
                        if isClassifier:
                            a = numpy.concatenate([[x.optimizer, datasets[j], x.objfname, k + 1,
                                                    x.startTime, x.endTime, x.executionTime,
                                                    x.trainAcc, x.trainTP, x.trainFN, x.trainFP,
                                                    x.trainTN, x.testAcc, x.testTP, x.testFN,
                                                    x.testFP, x.testTN], x.convergence])
                            writer.writerow(a)
                        else:
                            a = numpy.concatenate([[x.optimizer, datasets[j], x.objfname, k + 1,
                                                    x.startTime, x.endTime, x.executionTime,
                                                    x.trainAcc, x.trainTP, x.trainFN, x.trainFP,
                                                    x.trainTN, x.testAcc, x.testTP, x.testFN,
                                                    x.testFP, x.testTN, x.trainMSE, x.trainMAE,
                                                    x.trainRMSE, x.trainMMRE, x.trainVAF, x.trainED,
                                                    x.trainR2, x.testMSE, x.testMAE, x.testRMSE,
                                                    x.testMMRE, x.testVAF, x.testED, x.testR2,
                                                    x.layers], x.convergence])
                            writer.writerow(a)
                    out.close()
                Flag = True  # at least one experiment

if (Flag == False):  # Faild to run at least one experiment
    print(
        "No Optomizer or Cost function is selected. Check lists of available optimizers and cost functions")
