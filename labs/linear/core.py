import math
import numpy as np

class Dataset(object):
    def __init__(self, X, Y, attribute_number):
        self.X = X
        self.Y = Y
        self.attribute_number = attribute_number

def __init_weight(m):
    limit = 1.0 / (2.0 * m)
    return np.random.normal(-limit, limit, m)

def lose_function(X, W, Y):
    return np.sum((Y - X.dot(W)) ** 2) / Y.size

def __update_weight(X, Y, W):
    curr_diffs = X.dot(W) - Y

    Gr = X.T.dot(curr_diffs * 2)
    h = np.sum(curr_diffs / X.dot(Gr)) / Y.size

    return W - h * Gr

def gradient_descent_steps(dataset, max_iter):
    X = np.array(dataset.X)
    Y = np.array(dataset.Y)
    W = __init_weight(dataset.attribute_number)

    for _ in range(max_iter):
        W = __update_weight(X, Y, W)

    return W

def gradient_descent(dataset):
    X = np.array(dataset.X)
    Y = np.array(dataset.Y)
    W = __init_weight(dataset.attribute_number)

    q = lose_function(X, W, Y)
    while q > 0.01:
        W = __update_weight(X, Y, W)
        q = lose_function(X, W, Y)

    return W

def generalized_inverse(dataset):
    X = np.array(dataset.X)
    Y = np.array(dataset.Y)

    X_ = np.linalg.pinv(X)
    return np.matmul(X_, Y)
