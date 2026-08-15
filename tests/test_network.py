import sys
import unittest
from pathlib import Path

import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from MyNetwork import Network


class NetworkTests(unittest.TestCase):
    def setUp(self):
        np.random.seed(7)
        self.network = Network([2, 3, 2])
        self.x = np.array([[0.25], [-0.4]])
        self.y = np.array([[1.0], [0.0]])

    def loss(self):
        output = self.network.feedForward(self.x)[1][-1]
        return 0.5 * float(np.sum((output - self.y) ** 2))

    def test_forward_shapes(self):
        z, activations = self.network.feedForward(self.x)
        self.assertEqual([value.shape for value in z], [(3, 1), (2, 1)])
        self.assertEqual(
            [value.shape for value in activations],
            [(2, 1), (3, 1), (2, 1)],
        )

    def test_backprop_matches_finite_differences(self):
        z, activations = self.network.feedForward(self.x)
        analytic_w, analytic_b = self.network.backprop(activations, self.y, z)
        epsilon = 1e-6

        for layer, weight in enumerate(self.network.weights):
            for index in np.ndindex(weight.shape):
                original = weight[index]
                weight[index] = original + epsilon
                loss_plus = self.loss()
                weight[index] = original - epsilon
                loss_minus = self.loss()
                weight[index] = original
                numerical = (loss_plus - loss_minus) / (2 * epsilon)
                self.assertAlmostEqual(numerical, analytic_w[layer][index], places=6)

        for layer, bias in enumerate(self.network.biases):
            for index in np.ndindex(bias.shape):
                original = bias[index]
                bias[index] = original + epsilon
                loss_plus = self.loss()
                bias[index] = original - epsilon
                loss_minus = self.loss()
                bias[index] = original
                numerical = (loss_plus - loss_minus) / (2 * epsilon)
                self.assertAlmostEqual(numerical, analytic_b[layer][index], places=6)

    def test_partial_batch_uses_its_actual_size(self):
        z, activations = self.network.feedForward(self.x)
        expected_w, expected_b = self.network.backprop(activations, self.y, z)
        before_w = [weight.copy() for weight in self.network.weights]
        before_b = [bias.copy() for bias in self.network.biases]
        eta = 0.1

        self.network.calcNabla([[(self.x, self.y)]], eta, miniBatchSize=8)

        for actual, before, gradient in zip(self.network.weights, before_w, expected_w):
            np.testing.assert_allclose(actual, before - eta * gradient)
        for actual, before, gradient in zip(self.network.biases, before_b, expected_b):
            np.testing.assert_allclose(actual, before - eta * gradient)


if __name__ == "__main__":
    unittest.main()
