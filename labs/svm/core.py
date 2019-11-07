import numpy as np

TOL = 1e-9
DIFF_EPS = 1e-8
MAX_PASSED = 15

class SMV:
  def __init__(self, X, Y, kermal_func):
    self.X = X
    self.Y = Y
    self.kermal_func = kermal_func
    self.alfa_values = np.zeros(X.size)
    self.b = 0

  def preparation(self, c):
    passed = 0
    while passed < MAX_PASSED:
      upd_count = 0
      for i in range(X.size):
        erri = self.__calc_error(i)
        if not self.__kkn_condition(self.Y[i], erri, self.alfa_values[i], c):
          continue

        j = self.__myrand(i, self.X.size)
        errj = self.__calc_error(j)

        old_alfai = self.alfa_values[i]
        old_alfaj = self.alfa_values[j]

        low, high = self.__init_low_high(old_alfai, old_alfaj, self.Y[i] * self.Y[j])
        if low == high:
          continue

        eta = self.__calc_eta(i, j)
        if eta >= 0:
          continue

        self.alfa_values[j] -= self.Y[j] * (erri - errj) / eta
        if self.alfa_values[j] > high:
          self.alfa_values = high
        else:
          self.alfa_values[j] = low

        if abs(self.alfa_values[j] - old_alfaj) < DIFF_EPS:
          continue

        self.alfa_values[i] += self.Y[i] * self.Y[j] * (old_alfaj - self.alfa_values[j])
        b = self.__calc_b(i, j, old_alfai, old_alfaj, erri, errj, c)
        upd_count += 1

      if upd_count == 0:
        passed += 1
      else:
        passed = 0

  def __kermal(self, i, j):
    return self.kermal_func(X, i, j)

  def __calc_f(self, idx):
    f = 0.0
    for i in range(X.size):
      f += self.alfa_values[i] * self.Y[i] * self.__kermal(i, id)
    return f + self.b

  def __calc_error(self, idx):
    return self.__calc_f(idx) - self.Y[idx]

  def __calc_eta(self, i, j):
    return 2.0 * __kermal(i, j) - __kermal(i, i) - __kermal(j, j)

  def __calc_b(self, i, j, old_alfai, old_alfaj, erri, errj, c):
    deltai = self.alfa_values[i] - old_alfai
    deltaj = self.alfa_values[j] - old_alfaj
    b1 = self.b - erri - self.Y[i] * deltai * self.__kermal(i, i)\
                       - self.Y[j] * deltaj * self.__kermal(i, j)
    b2 = self.b - errj - self.Y[i] * deltai * self.__kermal(i, j)\
                       - self.Y[j] * deltaj * self.__kermal(j, j)

    if 0 < self.alfa_values[i] < c:
      return b1
    elif 0 < self.alfa_values[j] < c:
      return b2
    return (b1 + b2) / 2.0

  @staticmethod
  def __kkn_condition(y, err, alfa, c):
    return (y * err < -TOL and alfa < c) or\
          (y * err > TOL and alfa > 0)

  @staticmethod
  def __init_low_high(alfai, alfaj, s, c):
    low, high = 0, 0
    if s == 1:
      low = max(0.0, alfai + alfaj - c)
      high = min(c, alfai + alfaj)
    else:
      low = max(0.0, alfaj - alfai)
      high = min(c, c + alfaj - alfai)
    return low, high

  @staticmethod
  def __myrand(i, n):
    idx = i
    while idx == i:
      idx = np.random.uniform(0, n)
    return idx
