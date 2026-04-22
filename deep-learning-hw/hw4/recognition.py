import data
import numpy as np
import functions as fun

# hyperparameters
epochs = 1000
nodes_1 = 50
nodes_2 = 50
nodes_3 = 50
alpha = 0.01

# get training data
X, D = data.get_training_data()

# initialize weights
rg = np.random.default_rng(0)
W1 = 2 * rg.random((nodes_1, 25)) - 1
W2 = 2 * rg.random((nodes_2, nodes_1)) - 1
W3 = 2 * rg.random((nodes_3, nodes_2)) - 1
W4 = 2 * rg.random((5, nodes_3)) - 1

for epoch in range(epochs):
    # fun.deep_relu(X, D, W1, W2, W3, W4, alpha)
    fun.deep_dropout(X, D, W1, W2, W3, W4, alpha)


X_test = data.get_test_data(3)
fun.results(W1, W2, W3, W4, X)