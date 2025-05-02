import numpy as np
import random

class Gauss2d():
    def __init__(self, numDataPoints,offset, std, seed):
        np.random.seed(seed)
        random.seed(seed)
        self._numDataPoints = numDataPoints
        self._offset = offset
        self._std = std
        self._GenerateDataSet()
        self._SplitData()
        self._trainingPos = 0
        self._validationPos = 0
        self._testingPos = 0

    def _GenerateDataSet(self):
        self._dataSet = np.zeros((self._numDataPoints, 3, 1))
        for i in range(self._numDataPoints):
            if random.random() > 0.5:
                self._dataSet[i, 0, 0] = int(1)
                self._dataSet[i, 1 :] = np.random.normal(0 + self._offset, self._std, size=(2, 1))
            else:
                self._dataSet[i, 0, 0] = int(0)
                self._dataSet[i, 1:] = np.random.normal(0 - self._offset, self._std, size=(2, 1))

    def _SplitData(self):
        self.trainingSetSize = int(round(self._numDataPoints * 0.8))
        self.validationSetSize = int(round(self._numDataPoints * 0.1))
        self.testingSetSize = self._numDataPoints - self.trainingSetSize - self.validationSetSize
        self._trainingSet = self._dataSet[:self.trainingSetSize]
        self._validationSet = self._dataSet[self.trainingSetSize : self.trainingSetSize + self.validationSetSize]
        self._testingSet = self._dataSet[self.trainingSetSize + self.validationSetSize :]


    def ShuffleTrainingData(self):
        np.random.shuffle(self._trainingSet)
        self._trainingPos = 0
        self._validationPos = 0
        self._testingPos = 0

    def GenerateExample(self, set):
        if set == "train":
            x = self._trainingSet[self._trainingPos, 1:]
            y = self._trainingSet[self._trainingPos, 0, 0]
            self._trainingPos += 1
        elif set == "validate":
            x = self._validationSet[self._validationPos, 1:]
            y = self._validationSet[self._validationPos, 0, 0]
            self._validationPos += 1
        elif set == "test":
            x = self._testingSet[self._testingPos, 1:]
            y = self._testingSet[self._testingPos, 0, 0]
            self._testingPos += 1
        else:
            raise ValueError("Invalid set kind!")

        return x, y


