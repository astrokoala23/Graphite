import numpy as np

class Layer:
    def forward(self, inputs):
        print("Layer: This is an abstract class!")

    def backward(self, dvalues):
        print("Layer: This is an abstract class!")

# Learnable Layers

class DenseLayer(Layer):
    def __init__(self, n_inp, n_neurons):
        super().__init__()
        self.weights = 0.01 * np.random.randn(n_inp, n_neurons)
        self.biases = np.zeros((1, n_neurons))

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


# Activation Layer

# Rectified Linear Unit (ReLU)
class ReLU(Layer):
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backwards(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

# Leaky ReLU (fixes the dead neuron problem - search it up)
class LeakyReLU(Layer):
    def __init__(self, alpha=0.05):
        self.alpha = alpha

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.where(inputs > 0, inputs, inputs * self.alpha) # Instead of 0 for negative numbers, a small gradient defined by alpha

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] *= self.alpha # Derivative is 1 if input > 0, else alpha

# Sigmoid (I just might be biased towards this)
class Sigmoid(Layer):
    def forward(self, inputs):
        self.inputs = inputs
        # It's clipped because past 500
        # the function returns basically 1
        # and to prevent overflow
        self.output = 1 / (1 + np.exp(-np.clip(inputs, -500, 500))) 

    def backward(self, dvalues):
        self.dinputs = dvalues * self.output * (1 - self.output)

# Tanh (technically better than sigmoid)
class Tanh(Layer):
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.tanh(inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output ** 2)


# Neural Network

class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, X):
        current_input = X
        for layer in self.layers:
            layer.forward(current_input)
            current_input = layer.output
        return current_input

    def backward(self, loss_grad):
        current_grad = loss_grad
        for layer in reversed(self.layers):
            layer.backward(current_grad)
            current_grad = layer.dinputs

