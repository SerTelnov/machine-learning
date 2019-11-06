import numpy as np

TOL = 1e-9
DIFF_EPS = 1e-8
MAX_PASSED = 15

def __kermal(X, i, j):
  return 0

def __myrand(i, n):
  idx = i
  while idx == i:
    idx = np.random.uniform(0, n)
  return idx

def calc_f(idx, X, Y, alfa_values, b):
  f = 0.0
  for i in range(X.size):
    f += alfa_values[i] * Y[i] * __kermal(X, i, id)
  return f + b

def __calc_error(idx, X, Y, alfa_values, b):
  return calc_f(idx, X, Y, alfa_values, b) - Y[idx]

def __kkn_condition(y, err, alfa, c):
  return (y * err < -TOL and alfa < c) or\
         (y * err > TOL and alfa > 0)

def __init_low_high(alfai, alfaj, s, c):
  low, high = 0, 0
  if s == 1:
    low = max(0.0, alfai + alfaj - c)
    high = min(c, alfai + alfaj)
  else:
    low = max(0.0, alfaj - alfai)
    high = min(c, c + alfaj - alfai)
  return low, high

def __calc_eta(X, i, j):
  return 0

def __calc_b(i, j, X, Y, old_alfai, old_alfaj, erri, errj, c):
  # deltai =  
  return 0

def svm(X, Y, c):
  alfa_values = np.zeros(X.size)
  b = 0

  passed = 0
  while passed < MAX_PASSED:
    upd_count = 0
    for i in range(X.size):
      erri = __calc_error(i, X, Y, alfa_values, b)
      if not __kkn_condition(Y[i], erri, alfa_values[i], c):
        continue
      
      j = __myrand(i, X.size)
      errj = __calc_error(j, X, Y, alfa_values, b)

      old_alfai = alfa_values[i]
      old_alfaj = alfa_values[j]

      low, high = __init_low_high(old_alfai, old_alfaj, Y[i] * Y[j], c)
      if low == high:
        continue

      eta = __calc_eta(X, i, j)
      if eta >= 0:
        continue
      alfa_values[j] -= Y[j] * (erri - errj) / eta
      if alfa_values[j] > high:
        alfa_values = high
      else:
        alfa_values[j] = low
      
      if abs(alfa_values[j] - old_alfaj) < DIFF_EPS:
        continue

      alfa_values[i] += Y[i] * Y[j] * (old_alfaj - alfa_values[j])
      b = __calc_b(i, j, X, Y, old_alfai, old_alfaj, erri, errj, c)
      upd_count += 1
    
    if upd_count == 0:
      passed += 1
    else:
      passed = 0
