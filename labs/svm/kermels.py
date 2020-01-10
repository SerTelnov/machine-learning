import math
import numpy as np

def linear_kernel(c):
  def kernel(xi, xj):
    return np.dot(xi, xj.T) + c
  return kernel

def polynomial_kernel(d):
  def kernel(xi, xj):
    return np.dot(xi, xj.T) ** d
  return kernel

def gaussian_kernel(sigma):
  def kernel(xi, xj):
#     return math.exp(-sigma * np.sum((xi - xj) ** 2))
    return np.exp(-sigma * (np.linalg.norm(xi - xj) ** 2))
#     return np.exp(-np.sqrt(np.linalg.norm(xi - xj) ** 2 / (2 * sigma **  2)))
  return kernel
