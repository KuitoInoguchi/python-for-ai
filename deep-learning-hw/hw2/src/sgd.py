import numpy as np
import utils

# hyperparams
epochs = 475
alpha = 0.9
hidden_nodes = 3
beta = 0.9

# training data
X, D = utils.get_preset_training_data(2)

# random weight initialization
rg = np.random.default_rng(0)
W1 = 2 * rg.random((hidden_nodes, 3)) - 1
W2 = 2 * rg.random((1, hidden_nodes)) - 1



# training
for epoch in range(epochs):
    utils.back_prop_xor(W1, W2, X, D, alpha, beta)

# result
utils.print_predictions(X, D, "After training with SGD method", W1, W2)

