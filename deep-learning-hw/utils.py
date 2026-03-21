import numpy as np

def sigmoid(x):
    """activation function"""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(y):
    """derivative of activation function"""
    return y * (1 - y)

def _compute_weight_increment(W, x, d, alpha):
    v = W @ x.T
    y = sigmoid(v)
    e = d - y
    delta = e * sigmoid_derivative(y)
    return alpha * delta * x

def sgd(W, X, D, alpha):
    """SGD function for one epoch of training"""
    for i in range(len(X)):
        x = X[i:i + 1, :]  # select one row of input
        d = D[i]  # select the corresponding correct output

        dW = _compute_weight_increment(W, x, d, alpha)
        W += dW

def batch_sgd(W, X, D, alpha):
    """Batch SGD function for one epoch of training"""
    avg_dW = np.zeros_like(W)
    for i in range(len(X)):
        x = X[i:i + 1, :]  # select one row of input
        d = D[i]  # select the corresponding correct output
        dW = _compute_weight_increment(W, x, d, alpha)
        avg_dW += dW / len(X)
    W += avg_dW

def small_batch_sgd(W, X, D, alpha, batch_size=2):
    # extract current batch
    for i in range(0, len(X), batch_size):
        batch_X = X[i:i + batch_size, :]
        batch_D = D[i:i + batch_size]

        # compute current weight increment for the batch at hand
        batch_dW = np.zeros_like(W)
        for j in range(len(batch_X)):
            x = batch_X[j:j + 1, :]
            d = batch_D[j:j + 1]
            dW = _compute_weight_increment(W, x, d, alpha) # dW for the sample
            batch_dW += dW # accumulating sum

        batch_dW /= len(batch_X)
        W += batch_dW


def generate_training_data(n_samples=100):
    rg = np.random.default_rng(0)

    # generate random input data
    X = rg.random((n_samples, 3))

    # a simple correct output: if the first element of the input is
    # greater than 0.5, expect a 1; otherwise, 0.
    D = (X[:, 0] > 0.5).astype(np.int64)

    return X, D

def get_preset_training_data(id=1):
    X = np.array([
        [0, 0, 1],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ])
    if id == 1:
        D = np.array([0, 0, 1, 1])
    elif id == 2:
        D = np.array([0, 1, 1, 0])
    return X, D

def print_predictions(W, X, D, title="Predictions"):
    """Print predictions for each input sample.

    Args:
        W: Weight matrix
        X: Input data matrix
        D: Desired output vector
        title: Title for the output section
    """
    print(f"{title}: ")
    print(f"Weights: {W}")
    for i in range(len(X)):
        x = X[i:i + 1, :]
        v = np.dot(W, x.T)
        y = sigmoid(v)
        print(f"input: {X[i]} -> output: {y[0][0]:.4f} (desired: {D[i]})")
    print()

