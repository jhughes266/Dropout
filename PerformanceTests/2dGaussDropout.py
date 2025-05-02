import random

import numpy as np
from Layers.DenseLayer import DenseLayer
from Layers.CostLayer import MSECostLayer
from Layers.DropoutLayer import DropoutLayer
from DataProcessing.Gauss2d import Gauss2d
from Graphing.DynamicGrapher import DynamicGrapher
import random
seed = 2
#dataSetup
numDataPoints = 1000
offset = 1
std = 3
gauss2d = Gauss2d(numDataPoints, offset, std, seed)

#hyperparmeter setup
epochs = 100
examplesPerEpoch = gauss2d.trainingSetSize
batchSize = 10
batchesPerEpoch = int(examplesPerEpoch / batchSize)
lr = 0.1
dropoutRate = 0.2

#grapher set up
grapher = DynamicGrapher("Epoch", "Cost")

#layer set up
L1 = DenseLayer(2, 500, "Tanh", batchSize, lr, seed)
L1Drop = DropoutLayer(500, dropoutRate)
L2 = DenseLayer(500, 200, "Tanh", batchSize, lr, seed)
L2Drop = DropoutLayer(200, dropoutRate)
L3 = DenseLayer(200, 1, "Sigmoid", batchSize, lr, seed)
L4train = MSECostLayer(gauss2d.trainingSetSize)
L4validate = MSECostLayer(gauss2d.validationSetSize)
L4test = MSECostLayer(gauss2d.testingSetSize)

for epochNum in range(epochs):
    gauss2d.ShuffleTrainingData()

    #training
    for batchNum in range(batchesPerEpoch):
        L1Drop.GenerateDropoutMask()
        L2Drop.GenerateDropoutMask()
        for exampleNum in range(batchSize):
            x, y = gauss2d.GenerateExample("train")

            # forward
            temp = L1.Forward(x)
            temp = L1Drop.Forward(temp)
            temp = L2.Forward(temp)
            temp = L2Drop.Forward(temp)
            yhat = L3.Forward(temp)
            L4train.Forward(y, yhat)

            # backward
            dyhat = L4train.Backward()
            dtemp = L3.Backward(dyhat)
            dtemp = L2Drop.Backward(dtemp)
            dtemp = L2.Backward(dyhat)
            dtemp = L1Drop.Backward(dtemp)
            L1.Backward(dtemp)

        L3.GradientStep()
        L2.GradientStep()
        L1.GradientStep()
        L3.Reset()
        L2.Reset()
        L1.Reset()
    trainingCost = L4train.DisplayInfo(message=("Epoch: " + str(epochNum + 1) + " Training Metrics"), display=True)
    L4train.Reset()

    #validation
    for exampleNum in range(gauss2d.validationSetSize):
        x, y = gauss2d.GenerateExample('validate')
        temp = L1.Forward(x)
        temp = L2.Forward(temp)
        yhat = L3.Forward(temp)
        L4validate.Forward(y, yhat)
    validationCost = L4validate.DisplayInfo(message=("Epoch: " + str(epochNum + 1) + " Validation Metrics"),display=True)
    L4validate.Reset()
    L3.Reset()
    L2.Reset()
    L1.Reset()

    grapher.UpdatePlot(trainingCost, validationCost)
#testing
for testNum in range(gauss2d.testingSetSize):
    x, y = gauss2d.GenerateExample('test')
    temp = L1.Forward(x)
    temp = L2.Forward(temp)
    yhat = L3.Forward(temp)
    L4test.Forward(y, yhat)
L4test.DisplayInfo(message=("Testing Metrics"),display=True)
grapher.FinishPlot()





