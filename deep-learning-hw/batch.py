import numpy as np
import utils

# training data
# X, D = utils.generate_training_data(10) # X is (100, 3) by default
X, D = utils.get_preset_training_data()

# random weight initialization
rg = np.random.default_rng(0)
W = 2 * rg.random((1, 3)) - 1

# hyperparams
epochs = 4000
alpha = 0.3

# training
for epoch in range(epochs):
    utils.batch_sgd(W, X, D, alpha)

# result
utils.print_predictions(W, X, D, "After training")
