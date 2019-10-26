import numpy as np

Xtrain=np.loadtxt(open("Xtrain.csv","rb"),delimiter=",",skiprows=0)
Xtst=np.loadtxt(open("Xtst.csv","rb"),delimiter=",",skiprows=0)
Xtst1=np.loadtxt(open("Xtst1.csv","rb"),delimiter=",",skiprows=0)  
   
Ytrain=np.loadtxt(open("Ytrain.csv","rb"),delimiter=",",skiprows=0)
Ytst=np.loadtxt(open("Ytst.csv","rb"),delimiter=",",skiprows=0)
Ytst1=np.loadtxt(open("Ytst1.csv","rb"),delimiter=",",skiprows=0)

   
X_tot = np.concatenate((Xtrain,Xtst1), axis=0)


    #normalization for output
minX = X_tot.min( axis = 0)
maxX = X_tot.max( axis = 0)
normalizationFactor = (maxX-minX)
XtrainN = (Xtrain-minX)/normalizationFactor
Xtst1tN = (Xtst1-minX)/normalizationFactor

XY_train = np.concatenate((XtrainN,Ytrain.reshape((len(Ytrain),1))), axis=1)
XY_test = np.concatenate((Xtst1tN,Ytst1.reshape((len(Ytst1),1))), axis=1)


np.savetxt("PowerTrain1.csv", XY_train, delimiter=",")
np.savetxt("PowerTest1.csv", XY_test, delimiter=",")
