import numpy as np
import time as time
import random

class Network:


    def __init__(self, layers):
        self.layers = layers
        self.numLayers = len(layers)
        self.biases = initBias(layers)
        self.weights = initWeight(layers)

    def feedForward(self, x):
        """x is the vector of input neurons, y is the output vector"""

        z = initZ(self.layers)
        activation = initActivation(self.layers)
        activation[0] = x
        z[0] = np.dot(self.weights[0], x) + self.biases[0]
        activation[1] = sigmoid(z[0])

        for k in range(self.numLayers-1):
            #I have to calculate the output activation in this loop
            z[k] = np.dot(self.weights[k], activation[k]) + self.biases[k]
            activation[k+1] = sigmoid(z[k])

        # cost = self.calcCost(activation[self.numLayers - 1], y)

        # nablaList = [nabla_w, nabla_b]

        # nablaList = self.backprop(activation, y, z)
        output = [z, activation]
        return output

    def calcCost(self, a , y):
        diff = a-y
        return (diff**2)
    

    def backprop(self, activation, y, z,):
        nabla_w = initNabla_w(self.layers)
        nabla_b = initNabla_b(self.layers)
        delta = initDelta(self.layers)

        ##compute the last layer of errors
        delta[-1] = calcLastLayer(activation, y, z[-1])

        #complete the errors
        for i in range(self.numLayers - 3, -1, -1):
            delta[i] = (np.dot(self.weights[i+1].T, delta[i+1]) * sigmaPrime(z[i]))

        for i in range(0, self.numLayers-1):
            nabla_b[i] = delta[i]

        nabla_w = createNabla_w(nabla_w, delta, activation, self.numLayers)
        nablaList = [nabla_w, nabla_b]
        return nablaList
    
    def getMiniBatch(self, training_data, miniBatchSize):
        random.shuffle(training_data)
        miniBatches = [
            training_data[k : k + miniBatchSize] 
            for k in range(0, len(training_data), miniBatchSize)
        ]

        return miniBatches
    
    def calcNabla(self, miniBatches, eta, miniBatchSize ):
        
        for miniBatch in miniBatches:
            
            nabla_b = [np.zeros((b,1)) for b in self.layers[1:]]
            nabla_w = [np.zeros((n,m)) for m,n in zip(self.layers[:-1], self.layers[1:])]
            # counter = 0
            for x,y in miniBatch:
                # counter += 1
                neuronsAct = self.feedForward(x)
                nablaList = self.backprop(neuronsAct[1], y, neuronsAct[0])
                nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, nablaList[1])]
                nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, nablaList[0])]
            
            nabla_b = [nb/miniBatchSize for nb in nabla_b] 
            nabla_w = [nw/miniBatchSize for nw in nabla_w]

            self.update_parameters(nabla_w, nabla_b, eta)

    def update_parameters(self, nabla_w, nabla_b, eta):
        self.weights = [w - eta*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - eta*nb for b, nb in zip(self.biases, nabla_b)]

    def SGD(self, training_data, test_data, eta, epoch, miniBatchSize):
        
        for i in range(epoch):
            miniBatches = self.getMiniBatch(training_data, miniBatchSize)
            self.calcNabla(miniBatches, eta, miniBatchSize)
            self.eval(test_data, i)

    def eval(self, test_data, epoch):
        counter = 0
        for x,y in test_data:
            output  = np.argmax(self.feedForward(x)[1][-1])
            # output += 1
            if output == y:
                counter += 1
        print(f"{counter}/{len(test_data)} at epoch {epoch} ")



    


def initBias(layers):
    biases = [np.random.randn(b,1) for b in layers[1:]] #note that it should be [] to become a list, otherwise it becomes a generator (?)
    return biases

def initWeight(layers):
    weights = [np.random.randn(n,m) for m,n in zip(layers[:-1], layers[1:])]
    return weights

def initZ(layers):
    Zs = [np.random.randn(z,1) for z in layers[1:]] #note that it should be [] to become a list, otherwise it becomes a generator (?)
    return Zs

def initActivation(layers):
    activations = [np.random.randn(a,1) for a in layers[:]] #note that it should be [] to become a list, otherwise it becomes a generator (?)
    return activations

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def initNabla_w(layers):
    nabla_w = [np.random.randn(n,m) for m,n in zip(layers[:-1], layers[1:])]
    return nabla_w

def initNabla_b(layers):
    nabla_b = [np.random.randn(b,1) for b in layers[1:]] #note that it should be [] to become a list, otherwise it becomes a generator (?)
    return nabla_b

def initDelta(layers):
    delta = [np.random.randn(b,1) for b in layers[1:]] #note that it should be [] to become a list, otherwise it becomes a generator (?)
    return delta

def calcLastLayer(activation, y, z):
    return (activation[-1] - y) * sigmaPrime(z)

def sigmaPrime(z):
    return (np.exp(-z)) / ((1 + np.exp(-z))**2)

def createNabla_w(nabla_w, delta, activation, num):
    for i in range(num-1):
        nabla_w[i] = np.dot(delta[i], activation[i].T)

    return nabla_w

if __name__ == "__main__":
    import mnist_loader
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = Network([784, 30, 10])
    net.SGD(training_data, test_data, eta=3.0, epoch=30, miniBatchSize=10)