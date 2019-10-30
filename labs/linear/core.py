import math
import numpy as np


class Dataset(object):    
    def __init__(self, X, Y, attribute_number):
        self.X = X
        self.Y = Y
        self.attribute_number = attribute_number

def __init_weight__(m):
    limit = 1.0 / (2.0 * m)
    return np.random.uniform(low = -limit, high = limit, size = m)

def calc_answers(X, W):
    return X @ W

def lose_function(A, Y):
    return np.sum(np.power(Y - A, 2)) / Y.size


def __update_weight__(X, Y, W, attribute_number):
    curr_diffs = calc_answers(X, W) - Y

    Gr = np.sum(((X.T) * (curr_diffs * 2)).T, axis = 0)
    h = np.sum(curr_diffs / (X @ Gr)) / Y.size

    return W - h * np.true_divide(Gr, Y.size)


def gradient_descent_steps(dataset, max_iter):
    W = __init_weight__(dataset.attribute_number)

    X = np.array(dataset.X)
    Y = np.array(dataset.Y)

    for _ in range(max_iter):
        W = __update_weight__(X, Y, W, dataset.attribute_number)

    return W

def gradient_descent(dataset):
    W = __init_weight__(dataset.attribute_number)
    l = 0.1
    X = np.array(dataset.X)
    Y = np.array(dataset.Y)

    q = lose_function(calc_answers(X, W), Y)
    while q > 1.1:
        W = __update_weight__(X, Y, W, dataset.attribute_number)
        eplison = lose_function(calc_answers(X, W), Y)
        q = l * eplison + (1 - l) * q

    return W