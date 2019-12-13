import numpy as np

TOL = 1e-6
DIFF_EPS = 1e-5
MAX_PASSED = 10
MAX_ITER = 100

class SVM:
  def __init__(self, X, Y, kermel_func):
    self.X = X
    self.Y = Y
    self.K = self.__eval_kernel(kermel_func)
    self.alpha_values = np.zeros(len(self.X))
    self.b = 0

  def classify(self, X):
    if type(X) is not np.ndarray:
        X = np.array(X)
    return np.fromiter(map(lambda x: self.classify1(x), X), int)

  def classify1(self, x):
    value = np.sum(x * self.W) + self.b
    return 1 if value > 0 else -1

  def evaluate(self, c):
    passed = 0
    iter = 0
    while passed < MAX_PASSED and iter < MAX_ITER:
      upd_count = 0
      for i in range(len(self.X)):
        erri = self.__calc_error(i)
        if not self.__kkn_condition(self.Y[i], erri, self.alpha_values[i], c):
          continue

        j = self.__myrand(i, len(self.X))
        errj = self.__calc_error(j)

        old_alphai = self.alpha_values[i]
        old_alphaj = self.alpha_values[j]

        low, high = self.__init_low_high(old_alphai, old_alphaj, self.Y[i] * self.Y[j], c)
        if low == high:
          continue

        eta = self.__calc_eta(i, j)
        if eta >= 0:
          continue

        self.alpha_values[j] -= self.Y[j] * (erri - errj) / eta
        if self.alpha_values[j] > high:
          self.alpha_values[j] = high
        else:
          self.alpha_values[j] = low

        if abs(self.alpha_values[j] - old_alphaj) < DIFF_EPS:
          continue

        self.alpha_values[i] += self.Y[i] * self.Y[j] * (old_alphaj - self.alpha_values[j])
        self.b = self.__calc_b(i, j, old_alphai, old_alphaj, erri, errj, c)
        upd_count += 1

      iter += 1
      if upd_count == 0:
        passed += 1
      else:
        passed = 0

    self.W = self.__compute_w()
    return self

  def __eval_kernel(self, kermel_func):
    K = np.zeros((len(self.X), len(self.X)))
    for i in range(len(self.X)):
      for j in range(len(self.X)):
        K[i][j] = kermel_func(self.X[i,:], self.X[j,:])
    return K

  def __compute_w(self):
      m, n = np.shape(self.X)
      w = np.zeros(n)
      for i in range(m):
          w += np.multiply(self.alpha_values[i] * self.Y[i], self.X[i,:].T)
      return w

  def __kermal(self, i, j):
    return self.K[i][j]

  def __calc_f(self, idx):
    f = 0.0
    for i in range(len(self.X)):
      f += self.alpha_values[i] * self.Y[i] * self.__kermal(i, idx)
    return f + self.b

  def __calc_error(self, idx):
    return self.__calc_f(idx) - self.Y[idx]

  def __calc_eta(self, i, j):
    return 2.0 * self.__kermal(i, j) - self.__kermal(i, i) - self.__kermal(j, j)

  def __calc_b(self, i, j, old_alphai, old_alphaj, erri, errj, c):
    deltai = self.alpha_values[i] - old_alphai
    deltaj = self.alpha_values[j] - old_alphaj
    b1 = self.b - erri - self.Y[i] * deltai * self.__kermal(i, i)\
                       - self.Y[j] * deltaj * self.__kermal(i, j)
    b2 = self.b - errj - self.Y[i] * deltai * self.__kermal(i, j)\
                       - self.Y[j] * deltaj * self.__kermal(j, j)

    if 0 < self.alpha_values[i] < c:
      return b1
    elif 0 < self.alpha_values[j] < c:
      return b2
    return (b1 + b2) / 2.0

  @staticmethod
  def __kkn_condition(y, err, alpha, c):
    return (y * err < -TOL and alpha < c) or\
          (y * err > TOL and alpha > 0)

  @staticmethod
  def __init_low_high(alphai, alphaj, s, c):
    if s == 1:
      low = max(0.0, alphai + alphaj - c)
      high = min(c, alphai + alphaj)
    else:
      low = max(0.0, alphaj - alphai)
      high = min(c, c + alphaj - alphai)
    return low, high

  @staticmethod
  def __myrand(i, n):
    idx = i
    while idx == i:
      idx = np.random.randint(0, n)
    return idx

