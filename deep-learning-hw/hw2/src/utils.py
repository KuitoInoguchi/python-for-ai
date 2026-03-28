import numpy as np

def sigmoid(x):
    """activation function"""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(r_sigmoid):
    """
    derivative of sigmoid function, but, pass me the result of sigmoid!
    """
    return r_sigmoid * (1 - r_sigmoid)

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

def mini_batch_sgd(W, X, D, alpha, batch_size=2):
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

def back_prop_xor(W1, W2, X, D, alpha, beta=0.0):
    """Backpropagation function for XOR gate"""
    mmt1 = np.zeros_like(W1)
    mmt2 = np.zeros_like(W2)

    for i in range(len(X)):
        x = X[i:i + 1, :].T  # select one row of input
        d = D[i]  # select the corresponding correct output

        # the hidden layer
        v1 = W1 @ x
        y1 = sigmoid(v1)

        # the output layer
        v = W2 @ y1
        y = sigmoid(v)

        # delta of the output layer
        e = d - y
        delta = sigmoid_derivative(y) * e

        # delta of the hidden layer
        e1 = W2.T @ delta
        delta1 = sigmoid_derivative(y1) * e1

        # weight update
        dW1 = alpha * delta1 @ x.T
        mmt1 = dW1 + beta * mmt1
        W1 += mmt1

        dW2 = alpha * delta @ y1.T
        mmt2 = dW2 + beta * mmt2
        W2 += mmt2


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
    else:
        raise ValueError("Invalid id. Please choose 1 or 2.")
    return X, D

def print_predictions(X, D, title="Predictions", *weights):
    """Print predictions for each input sample with arbitrary number of weight matrices.

    Args:
        X: Input data matrix
        D: Desired output vector
        title: Title for the output section
        *weights: Variable number of weight matrices for multi-layer networks
    """
    print(f"{title}: ")
    
    if len(weights) == 0:
        print("No weights provided.")
        return
    elif len(weights) == 1:
        # Single layer network
        W = weights[0]
        print(f"Weights: {W}")
        for i in range(len(X)):
            x = X[i:i + 1, :]
            v = np.dot(W, x.T)
            y = sigmoid(v)
            print(f"input: {X[i]} -> output: {y[0][0]:.4f} (desired: {D[i]})")
    else:
        # Multi-layer network with multiple weight matrices
        for idx, W in enumerate(weights):
            print(f"Weight Matrix {idx + 1}: \n{W}")
        
        for i in range(len(X)):
            # Start with input
            current_input = X[i:i + 1, :].T  # Transpose input
            
            # Propagate through all layers
            for W in weights:
                v = W @ current_input
                current_input = sigmoid(v)  # Output of current layer becomes input for next
            
            # Final output after passing through all layers
            y = current_input
            print(f"input: {X[i]} -> output: {y[0][0]:.4f} (desired: {D[i]})")
    print()