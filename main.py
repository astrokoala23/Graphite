from nn import *
import numpy as np

# Dummy training data
X = np.array([[1.0, 3.0], [2.5, 7.5], [2.1, -0.5]])
y_true = np.array([[1.0], [2.0], [0.5]])

nn = NeuralNetwork()
nn.add(DenseLayer(2, 10))
nn.add(LeakyReLU())
nn.add(DenseLayer(10, 1))

# Train
nn.fit(X, y_true, 100000)

print(f"Final Predictions: \n{nn.forward(X)}")