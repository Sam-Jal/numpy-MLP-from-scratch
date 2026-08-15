# NumPy MLP from Scratch

A multilayer perceptron implemented with NumPy and trained on the MNIST
handwritten-digit dataset. The neural-network operations—forward propagation,
quadratic loss, backpropagation, mini-batch SGD, and parameter updates—are
implemented without PyTorch, TensorFlow, or an automatic-differentiation
library.

The project was built to make the mathematics of a small neural network
explicit and testable.

## Architecture

```text
784 input neurons
        ↓
30 sigmoid hidden neurons
        ↓
10 sigmoid output neurons
```

The output class is the index of the largest output activation.

## Implemented components

- random initialization of weights and biases;
- feed-forward propagation;
- sigmoid activations and their derivatives;
- quadratic loss;
- backpropagation;
- mini-batch stochastic gradient descent;
- parameter updates and classification-accuracy reporting.

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python code/download_mnist.py
```

The download script stores `mnist.pkl.gz` under `data/`, which is excluded
from Git.

## Train on MNIST

From the repository root, run:

```bash
python code/MyNetwork.py
```

The default experiment trains a `784-30-10` network for 30 epochs using a
learning rate of `3.0` and mini-batches of 10. After every epoch, the script
prints the number of correctly classified test examples.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests include a numerical finite-difference gradient check. This verifies
that the analytical backpropagation gradients match the change in loss caused
by perturbing every weight and bias in a small network.

## Project structure

```text
code/MyNetwork.py       neural-network implementation and training entry point
code/mnist_loader.py    MNIST deserialization and preprocessing
code/download_mnist.py  dataset download helper
tests/test_network.py   shape, gradient, and mini-batch tests
```

## Attribution

`mnist_loader.py` and the serialized MNIST dataset format are based on the
examples accompanying Michael Nielsen's *Neural Networks and Deep Learning*.
The network implementation in `MyNetwork.py` was written separately for this
project.
