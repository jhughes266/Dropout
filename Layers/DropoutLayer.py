import numpy as np
import random
class DropoutLayer():
    def __init__(self, shape, dropoutRate):
        self._shape = shape
        self._dropoutRate = dropoutRate
        self._scaleFactor = 1 / (1 - dropoutRate)

    def GenerateDropoutMask(self):
        self._mask = np.ones((self._shape, 1))
        for i in range(self._shape):
            if random.random() < self._dropoutRate:
                self._mask[i, 0] = 0

    def Forward(self, x):
        y = self._scaleFactor * self._mask * x
        return y

    def Backward(self, dy):
        dx = dy * self._mask * self._scaleFactor
        return dx