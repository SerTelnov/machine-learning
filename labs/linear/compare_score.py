import numpy as np
import math

import core
from data import read_data

MAX_ITER = 1000

TEST_DATA_PATH = "labs/linear/resources/"

def __calc_nrmse__(Y, A):
    return math.sqrt(\
        np.sum(\
            np.power(Y - A, 2)\
        ) / Y.size)

for test_number in range(0, 7):
    print("start for dataset #" + str(test_number))

    train, test = read_data(TEST_DATA_PATH + str(test_number) + ".txt")
    print("read data")

    W = core.gradient_descent_steps(train, MAX_ITER)
    print("evaluated W by gradient descent")

    X = np.array(test.X)
    Y = np.array(test.Y)
    A = core.calc_answers(X, W)
    print("for gradient descent NRMSE: '" + str(__calc_nrmse__(Y, A)) + "'")
