# NumPy MLP from Scratch

A simple multilayer perceptron implemented from scratch using NumPy and
trained on the MNIST handwritten-digit dataset.

The purpose of this project was to understand the internal mechanics of a
neural network without relying on deep-learning frameworks such as PyTorch
or TensorFlow.

## Implemented components

The neural-network implementation includes:

- random initialization of weights and biases;
- feed-forward propagation;
- sigmoid activation functions;
- quadratic loss;
- backpropagation;
- mini-batch stochastic gradient descent;
- parameter updates;
- classification accuracy evaluation.

## Model architecture

The current model uses the following architecture:

```text
784 input neurons
        ↓
30 hidden neurons
        ↓
10 output neurons