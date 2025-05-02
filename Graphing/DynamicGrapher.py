import matplotlib.pyplot as plt

#reference: https://www.youtube.com/watch?v=7RgoHTMbp4A provided the neccessary information to build this
#also take away the color for a good experience
class DynamicGrapher:
    def __init__(self, xTitle, yTitle):
        #set up the axis titles
        self._xTitle = xTitle
        self._yTitle = yTitle
        #set up lists to store data to be graphed
        self._timeAxis = []
        self._trainingCost = []
        self._validationCost = []
        #keeps track of the time step
        self._t = 1

    def UpdatePlot(self, trainingCost, validationCost):

        #append new data into data arrays
        self._timeAxis.append(self._t)
        self._trainingCost.append(trainingCost)
        self._validationCost.append(validationCost)
        #setting the plt axis limits
        if self._t % 10 == 0:
            plt.xlim(0, self._t + 1)
            plt.ylim(0, max([max(self._trainingCost), max(self._validationCost)]) + .05)
            # setting titles
            plt.xlabel(self._xTitle)
            plt.ylabel(self._yTitle)
            # plotting the plot
            plt.plot(self._timeAxis, self._trainingCost, color='blue', label="training")
            plt.plot(self._timeAxis, self._validationCost, color='red', label="validation")

            # pause the plot
            plt.pause(0.000001)
        #increment the t
        self._t += 1

    #reuired to keep plot on screen once execution of program has finished
    def FinishPlot(self):
        plt.plot(self._timeAxis, self._trainingCost, color='blue', label="training")
        plt.plot(self._timeAxis, self._validationCost, color='red', label="validation")
        plt.show()