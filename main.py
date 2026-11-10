import numpy as np

class Layer:
    def forward(self, inputs):
        print("Layer: This is an abstract class!")

    def backward(self, dvalues):
        print("Layer: This is an abstract class!")

class DenseLayer(Layer):
    def __init__(self, n_inp, n_neurons):
        super().__init__()
        self.weights = 0.01 * np.random.randn(n_inp, n_neurons)
        self.biases = np.zeroes((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs

        # Matrix multiplication is way faster than calculating each individual neuron
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        # Alot of calculus that took way to long to understand
        self.dweights = np.dot(self.inputs.T, dvalues)
        # keepdims is used to maintain the shape of the array
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)