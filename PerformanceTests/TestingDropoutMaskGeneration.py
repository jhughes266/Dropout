import numpy as np
from Layers.DropoutLayer import DropoutLayer
test = DropoutLayer(10, 0.5)
sum = 0
for i in range(100):
    test.GenerateDropoutMask()
    print(test._mask)
    sum += np.count_nonzero(test._mask == 0)
    print("#################")
print(sum/ 100)