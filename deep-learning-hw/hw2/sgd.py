import numpy as np
import utils

# training data
X, D = utils.get_preset_training_data(2)

# random weight initialization
rg = np.random.default_rng(0)
W1 = 2 * rg.random((4, 3)) - 1
W2 = 2 * rg.random((1, 4)) - 1

# hyperparams
epochs = 10000
alpha = 0.1

# training
for epoch in range(epochs):
    utils.back_prop_xor(W1, W2, X, D, alpha)

# result
utils.print_predictions(X, D, "After training with SGD method", W1, W2)

