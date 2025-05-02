import numpy as np
class NoiseData():
    def __init__(self, shape):
        self._shape = shape

    def TrainGeneratorData(self):
        noiseVec = np.random.normal(0, 1, (self._shape, 1))
        y = np.ones((1,1))
        return noiseVec, y

    def TrainDiscriminatorData(self):
        noiseVec = np.random.normal(0, 1, (self._shape, 1))
        y = np.zeros((1,1))
        return noiseVec, y

