from DataProcessing.Mnist import Mnist
from Layers.CostLayer import MSECostLayer
from Layers.DenseLayer import DenseLayer
from Graphing.DynamicGrapher import DynamicGrapher
import numpy as np
from DataProcessing.MnistTest import MnistTest
import sys
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=np.inf)

epochs = 1
examplesPerEpoch = 50000
batchSize = 10
batchesPerEpoch = int(examplesPerEpoch / batchSize)
lr = 0.1
#data set up
mnist = Mnist()
mnist.LoadTrainMatrix()

mnistTest = MnistTest()
mnistTest.LoadTestMatrix()

#grapher set up
grapher = DynamicGrapher("Batch Number", "Loss")


L1 = DenseLayer(784, 1024, "Tanh", batchSize, lr)
L2 = DenseLayer(1024, 10, "Sigmoid", batchSize, lr)
L3 = MSECostLayer(batchSize)
testCost = MSECostLayer(mnistTest.numTestExamples)

imageCounter = 0

for epochNum in range(epochs):
    print("EPOCH:" +str(epochNum + 1))
    mnist.Shuffle()
    for batchNum in range(batchesPerEpoch):
        for exampleNum in range(batchSize):
            x, y = mnist.GenerateExample()


            #forward
            temp = L1.Forward(x)
            yhat = L2.Forward(temp)
            L3.Forward(y, yhat)

            #backward
            dyhat = L3.Backward()
            dtemp = L2.Backward(dyhat)
            L1.Backward(dtemp)


        graphData = L3.DisplayInfo(True)
        grapher.UpdatePlot(graphData,0)#no validation being used
        L2.GradientStep()
        L1.GradientStep()



        L3.Reset()
        L2.Reset()
        L1.Reset()
grapher.FinishPlot()

for i in range(mnistTest.numTestExamples):
    x, y = mnistTest.GenerateExample()
    temp = L1.Forward(x)

    if i % 50 == 0:
        plt.clf()
        plt.imshow(temp.reshape(32, 32), cmap="gray", vmin=-0.1, vmax=0.1)
        plt.colorbar()
        plt.draw()
        plt.pause(0.00001)
        imageCounter += 1

    yhat = L2.Forward(temp)
    testCost.Forward(y, yhat)
    testCost.DisplayInfo()
plt.show()




