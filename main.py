from nn import *
import numpy as np

# Dummy training data
X = np.array([[1.0, 3.0], [2.5, 7.5], [2.1, -0.5]])
y_true = np.array([[4.0], [-1.4], [-2.0]])

nn = NeuralNetwork()
nn.add(DenseLayer(2, 10))
nn.add(LeakyReLU())
nn.add(DenseLayer(10, 1))  # Fixed shape to match previous layer's output (10)

# Forward pass before training
pred = nn.forward(X)
mse = np.mean((pred - y_true) ** 2)
print(f"\nPrediction before training:\n{pred}")
print(f"Loss before training: {mse:.4f}")

# Training Loop (e.g., 100 epochs)
learning_rate = 0.01

for epoch in range(10000):
    # Forward pass
    pred = nn.forward(X)
    
    # Compute Loss & Loss Gradient
    loss_grad = 2 * (pred - y_true) / len(X)
    
    # Backward pass (computes gradients)
    nn.backward(loss_grad)
    
    # Update weights and biases using Gradient Descent
    for layer in nn.layers:
        # Only update layers that actually have weights (Dense layers do, Activations don't)
        if hasattr(layer, 'weights'):
            layer.weights -= learning_rate * layer.dweights
            layer.biases -= learning_rate * layer.dbiases

# Forward pass after training
final_pred = nn.forward(X)
final_mse = np.mean((final_pred - y_true) ** 2)
print(f"\nPrediction after training:\n{final_pred}")
print(f"Loss after training: {final_mse:.4f}")