import data
import numpy as np
import functions as fun

# hyperparameters
epochs = 1000
nodes = 50
alpha = 0.9

# get training data
X, D = data.get_training_data()

# initialize weights
rg = np.random.default_rng(0)
W1 = 2 * rg.random((nodes, 25)) - 1
W2 = 2 * rg.random((5, nodes)) - 1

for epoch in range(epochs):
    fun.multi_class(W1, W2, X, D, alpha)

fun.results(W1, W2, X)
