import numpy as np

class MSECostLayer():
    def __init__(self, epochSize):
        self._epochSize = epochSize
        self._yStore = 0
        self._yhatStore = 0
        self._costSum = 0
        self._correctSum = 0

    @property
    def CostSum(self):
        return self._costSum

    @property
    def CorrectSum(self):
        return self._correctSum

    def _IsCorrect(self, y, yhat):
        if np.argmax(y.flatten()) == np.argmax(yhat.flatten()):
            return 1
        return 0

    def _IsCorrectBinary(self, y, yhat):
        if np.all(np.round(yhat) == y):
            return 1
        return 0

    def Forward(self, y, yhat):
        #calculate
        self._costSum +=  np.sum(0.5 * np.square(y - yhat))
        #check if is correct
        self._correctSum += self._IsCorrectBinary(y, yhat)
        #store
        self._yStore = y
        self._yhatStore = yhat

    def Backward(self):
        dyhat = self._yhatStore - self._yStore
        return dyhat

    #This display may only be used in the testing of the reccurent cell
    def DisplayInfo(self, message='', display=True):
        if display:
            print(message)
            print("Sum Cost: " + str(self._costSum / self._epochSize))
            print("Correct Percentage: " + str( 100 * self._correctSum / self._epochSize) + "%")
            print("######################")
            return self._costSum / self._epochSize

    def Reset(self):
        self._costSum = 0
        self._correctSum = 0





