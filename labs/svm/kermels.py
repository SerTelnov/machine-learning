import math
import numpy as np

def linear_kernel(c, x, y):
    return np.dot(x, y) + c

def polynomial_kernel(d, x, y):
    return (np.dot(x, y) + 1) ** d

def gaussian_kernel(radial, x, y):
    return math.exp(-radial * (np.linalg.norm(x - y) ** 2))
