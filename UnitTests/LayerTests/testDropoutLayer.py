import unittest
import numpy as np
import numpy.testing as np_testing
from Layers.DenseLayer import DenseLayer
from Layers.DropoutLayer import DropoutLayer
class DropoutLayerTest(unittest.TestCase):
    def setUp(self):
        self.denseLayerTest = DenseLayer(2, 3, "Tanh", 10, 0.1)
        self.dropoutLayerTest = DropoutLayer(3, 0.5)
        self.denseLayerTest._w = np.array([0.2, -0.3, 0.05, 0.1, -0.5, 0.12]).reshape(3, 2)
        self.denseLayerTest._b = np.array([0.4, 1.2, -0.5]).reshape(3, 1)
        self.dropoutLayerTest._mask = np.array([0, 1, 0]).reshape(3, 1)

    def testForward(self):
        x = np.array([0.4, 0.6]).reshape(2, 1)
        denseOut = self.denseLayerTest.Forward(x)
        actualY = self.dropoutLayerTest.Forward(denseOut)
        desiredY = np.array([0, 1.712969831, 0]).reshape(3, 1)
        np_testing.assert_almost_equal(actualY, desiredY)

    def testBackward(self):
        x = np.array([0.4, 0.6]).reshape(2, 1)
        denseOut = self.denseLayerTest.Forward(x)
        self.dropoutLayerTest.Forward(denseOut)


        dy = np.array([0.34, -0.43, 0.12]).reshape(3, 1)
        desiredDDropoutLayerInput = np.array([0, -0.86, 0]).reshape(3, 1)
        actualDDropoutLayerInput = self.dropoutLayerTest.Backward(dy)
        np_testing.assert_almost_equal(actualDDropoutLayerInput, desiredDDropoutLayerInput)

        actualdx = self.denseLayerTest.Backward(actualDDropoutLayerInput)
        actualdw = self.denseLayerTest._dw
        actualdb = self.denseLayerTest._db

        desireddx = np.array([-0.011456644, -0.022913289]).reshape(2, 1)
        desireddw = np.array([0, 0, -0.091653155, -0.137479732, 0, 0]).reshape(3, 2)
        desireddb = np.array([0, -0.229132887, 0]).reshape(3, 1)

        np_testing.assert_almost_equal(actualdx, desireddx)
        np_testing.assert_almost_equal(actualdw, desireddw)
        np_testing.assert_almost_equal(actualdb, desireddb)



