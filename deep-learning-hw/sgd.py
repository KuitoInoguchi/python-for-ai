import numpy as np
import utils

# training data
X, D = utils.get_preset_training_data(2)

# random weight initialization
rg = np.random.default_rng(0)
W = 2 * rg.random((1, 3)) - 1

# hyperparams
epochs = 100
alpha = 0.9

# training
for epoch in range(epochs):
    utils.sgd(W, X, D, alpha)

# result
utils.print_predictions(W, X, D, "After training with SGD method")
