import unittest
import numpy as np
import numpy.testing as np_testing
from Layers.DenseLayer import DenseLayer
from Layers.CostLayer import MSECostLayer

class FullMiniBatchTest(unittest.TestCase):
    def setUp(self):
        batchSize = 3
        lr = 0.25
        #layer 1 setup
        self.L1 = DenseLayer(2, 3, "Sigmoid", batchSize, lr)
        self.L1._w = np.array([0.1, -0.67, 0.02, 0.9, -0.45, -0.34]).reshape(3, 2)
        self.L1._b = np.array([0.354, 0.53, 0.123]).reshape(3, 1)

        # layer 2 setup
        self.L2 = DenseLayer(3, 2, "Tanh", batchSize, lr)
        self.L2._w = np.array([0.23, 0.12, 0.56, -0.65, -0.8, 0.8]).reshape(2, 3)
        self.L2._b = np.array([0.345, 0.563]).reshape(2, 1)

        # layer 3 setup
        self.L3 = DenseLayer(2, 3, "Sigmoid", batchSize, lr)
        self.L3._w = np.array([0.4, 0.5, -0.23, 0.67, 0.97, -0.56]).reshape(3, 2)
        self.L3._b = np.array([0.345, 0.4, 0.2]).reshape(3, 1)

        #layer 4 (cost layer) setup
        self.L4 = MSECostLayer(batchSize)

    def testForwardBackward(self):
        x1 = np.array([0.56, 0.3]).reshape(2, 1)
        x2 = np.array([0.45, 0.87]).reshape(2, 1)
        x3 = np.array([0.45, 0.57]).reshape(2, 1)

        y1 = np.array([1, 0, 1]).reshape(3, 1)
        y2 = np.array([1, 0, 0]).reshape(3, 1)
        y3 = np.array([0, 0, 1]).reshape(3, 1)

        # 1
        temp = self.L1.Forward(x1)
        temp = self.L2.Forward(temp)
        yhat1 = self.L3.Forward(temp)
        self.L4.Forward(y1, yhat1)

        dtemp = self.L4.Backward()
        dtemp = self.L3.Backward(dtemp)
        dtemp = self.L2.Backward(dtemp)
        self.L1.Backward(dtemp)

        # 2
        temp = self.L1.Forward(x2)
        temp = self.L2.Forward(temp)
        yhat1 = self.L3.Forward(temp)
        self.L4.Forward(y2, yhat1)

        dtemp = self.L4.Backward()
        dtemp = self.L3.Backward(dtemp)
        dtemp = self.L2.Backward(dtemp)
        self.L1.Backward(dtemp)

        # 3
        temp = self.L1.Forward(x3)
        temp = self.L2.Forward(temp)
        yhat1 = self.L3.Forward(temp)
        self.L4.Forward(y3, yhat1)

        dtemp = self.L4.Backward()
        dtemp = self.L3.Backward(dtemp)
        dtemp = self.L2.Backward(dtemp)
        self.L1.Backward(dtemp)

        actualCostSum = self.L4.CostSum
        desiredCostSum = 1.139545443
        np_testing.assert_almost_equal(actualCostSum, desiredCostSum)


        self.L3.GradientStep()
        self.L2.GradientStep()
        self.L1.GradientStep()
        actualL1UpdatedW = self.L1._w
        actualL1UpdatedB = self.L1._b
        desiredL1UpdatedW = np.array([0.101829135, -0.668492412, 0.021770205, 0.901542493, -0.451749783, -0.341953478]).reshape(3, 2)
        desiredL1UpdatedB = np.array([0.357695443, 0.533591643, 0.11926955]).reshape(3, 1)
        np_testing.assert_almost_equal(actualL1UpdatedW, desiredL1UpdatedW)
        np_testing.assert_almost_equal(actualL1UpdatedB, desiredL1UpdatedB)

        actualL2UpdatedW = self.L2._w
        actualL2UpdatedB = self.L2._b
        desiredL2UpdatedW = np.array([0.23229734, 0.122224315, 0.561722697, -0.661337261, -0.815470045, 0.790557026]).reshape(2, 3)
        desiredL2UpdatedB = np.array([0.348629941, 0.541473288]).reshape(2, 1)
        np_testing.assert_almost_equal(actualL2UpdatedW, desiredL2UpdatedW)
        np_testing.assert_almost_equal(actualL2UpdatedB, desiredL2UpdatedB)

        actualL3UpdatedW = self.L3._w
        actualL3UpdatedB = self.L3._b
        desiredL3UpdatedW = np.array([0.400760784, 0.499916433, -0.252671652, 0.670523536, 0.969036878, -0.559575492]).reshape(3, 2)
        desiredL3UpdatedB = np.array([0.346208102, 0.365534625, 0.198281695]).reshape(3, 1)
        np_testing.assert_almost_equal(actualL3UpdatedW, desiredL3UpdatedW)
        np_testing.assert_almost_equal(actualL3UpdatedB, desiredL3UpdatedB)
