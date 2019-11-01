import numpy as np
import math

import core
from data import read_data

TEST_DATA_PATH = "labs/linear/resources/"

def calc_nrmse(X, Y, W):
  sum = 0

  for i in range(Y.size):
      sum += (Y[i] - (X[i] @ W)) ** 2

  y_diff = np.max(Y) - np.min(Y)
  return math.sqrt(sum / Y.size) / y_diff

def evaluate_nrmse_score(dataset, W):
  X = np.array(dataset.X)
  Y = np.array(dataset.Y)

  return calc_nrmse(X, Y, W)

for i in range(1, 8):
  print("Dataset #" + str(i))

  train, test = read_data(TEST_DATA_PATH + str(i) + ".txt")
  gd_W = core.gradient_descent(train)
  gd_score = evaluate_nrmse_score(test, gd_W)
  print("for gradient descent NRMSE: '" + str(gd_score) + "'")

  gi_W = core.generalized_inverse(train)
  gi_score = evaluate_nrmse_score(test, gi_W)
  print("for generalized inverse NRMSE: '" + str(gi_score) + "'")
