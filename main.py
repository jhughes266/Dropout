from DataProcessing.Gauss2d import Gauss2d
test = Gauss2d(10, 1, 1)
print(test._trainingSet)
#test.ShuffleTrainingData()
print("##########")
for i in range(test.trainingSetSize):
    print(test.GenerateExample("train"))
