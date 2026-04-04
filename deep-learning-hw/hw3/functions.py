import numpy as np
import utils

def multi_class(W1, W2, X, D, alpha):
    """ one epoch of training """
    for i in range(len(X)):
        x = X[i, :, :].reshape(1, -1).T ## x is of shape (25, 1)
        d = D[i] # d: scalar

        v1 = W1 @ x # W1: (nodes, 25)
        y1 = utils.sigmoid(v1) # y1: (nodes, 1)

        v = W2 @ y1 # v: (5, 1)
        y = utils.softmax(v) # y: (5, 1)
        e = d - y # e: (5, 1)
        delta = e # delta: (5, 1)

        e1 = W2.T @ delta
        delta1 = utils.sigmoid_derivative(y1) * e1

        W1 += alpha * delta1 @ x.T
        W2 += alpha * delta @ y1.T

def results(W1, W2, X):
    for i in range(len(X)):
        x = X[i, :, :].reshape(1, -1).T ## x is of shape (25, 1)
        v1 = W1 @ x # W1: (nodes, 25)
        y1 = utils.sigmoid(v1) # y1: (nodes, 1)

        v = W2 @ y1 # v: (5, 1)
        y = utils.softmax(v) # y: (5, 1)
        print(y)