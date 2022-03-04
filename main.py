import nnfs
from nnfs.datasets import spiral_data
import numpy as np
import matplotlib.pyplot as plt

# sets the random seed to 0
nnfs.init()

X, y = spiral_data(samples=100, classes=3)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
plt.show()

# ie fully connected layers
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # initialize using weights and biases

        # produces a Gaussian distribution with 0 mean, 1 var
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases =  np.zeros((1, n_neurons))


    def forward(self, inputs):
        # calculate output vals from inputs, weights, biases
        self.output = np.dot(inputs, self.weights) + self.biases

# create dataset with 100 samples, 3 classes
X, y = spiral_data(samples=100, classes=3)

# Layer with 2 input and 3 output
dense1 = Layer_Dense(2, 3)

# Forward pass of data through the Layer
dense1.forward(X)

print(dense1.output[:5])

