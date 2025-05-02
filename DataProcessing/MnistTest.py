import numpy as np
import pandas as pd
import random

class MnistTest():
    def __init__(self):
        self._mnistTestMatrix = None
        self._testFilePath = r"..\DataSets\mnist_test.csv"
        self._testPos = 0

    def LoadTestMatrix(self):
        df = pd.read_csv(self._testFilePath)
        self._mnistTestMatrix = df.values
        self.numTestExamples = self._mnistTestMatrix.shape[0]


    def GenerateExample(self):
        example = self._mnistTestMatrix[self._testPos]
        x = ((example[1:].reshape(784, 1)) - (255/2)) * (1/(255/2)) #* (1/ 255)
        y = np.zeros((10, 1))
        y[int(example[0])] = 1
        self._testPos += 1
        return x, y





