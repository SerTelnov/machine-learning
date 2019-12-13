import numpy as np
import math
from enum import Enum
from random import randint, shuffle

class Question:
  def __init__(self, feature_idx, value):
    self.feature_idx = feature_idx
    self.value = value

class Tree:
  def __init__(self, significance, left_cls, right_cls, question):
    self.significance = significance
    self.question = question
    self.left_cls = left_cls
    self.right_cls = right_cls

  def classify(self, row):
    clazz = self.left_cls if self._accept(row) else self.right_cls
    return clazz, self.significance

  def _accept(self, row):
    return row[self.question.feature_idx] < self.question.value

class AdaBoost:
  def __init__(self, models_number):
    self.models_number = models_number
    self.class_quantity = 2
    self.forest = []

  def fit(self, X, Y):
    self.X = X
    self.Y = Y
    self.W = self._init_weight()
    self.indices = np.arange(len(self.Y))

    for _ in range(self.models_number):
      self.next_model()

  def next_model(self):
      q = self._get_question(self.indices)
      left, right = self._split_on_classes(self.indices, q)
      significance, indices_with_error = self._calc_significance(self.W, self.indices, q, left, right)
      self.forest.append(Tree(significance, left, right, q))

      self.W = self._upd_weights(self.W, significance, len(self.indices), indices_with_error)
      self.indices = self._chooce_new_indices(self.W, len(self.indices))

  def predict(self, X):
    return np.fromiter(map(lambda x: self.classify(x), X), int)

  def classify(self, x):
    classes_score = np.zeros(2)
    for tree in self.forest:
      clazz, score = tree.classify(x)
      classes_score[1 if clazz == 1 else 0] += score
    return 1 if np.argmax(classes_score) == 1 else -1

  def _chooce_new_indices(self, W, n):
    indices = np.zeros(n, dtype=int)

    for i in range(n):
      random_value = np.random.uniform(low=0.0, high=1.0)
      for idx in range(n):
        random_value -= W[idx]
        if random_value <= 0:
          indices[i] = idx
          break
    return indices

  def _upd_weights(self, W, significance, n, indices_with_error):
    for i in range(n):
      v = significance if i in indices_with_error else -significance
      W[i] = W[i] * math.exp(v)

    x = sum(W)
    for i in range(n):
      W[i] = W[i] / x
    return W

  def _init_weight(self):
    a = np.empty(len(self.Y))
    a.fill(1 / len(self.Y))
    return a

  def _calc_significance(self, W, indices, q, left, right):
    total_error = 0
    indices_with_error = set()
    for idx, i in enumerate(indices):
      clazz = self.Y[i]
      actual = left if self._accept(self.X[i], q.feature_idx, q.value) else right

      if actual != clazz:
        total_error += W[idx]
        indices_with_error.add(idx)
    if total_error == 0:
      return np.inf, indices_with_error
    return math.log((1 - total_error) / total_error) / 2, indices_with_error

  def _get_question(self, indices):
    gini_winner = 2e9

    for feature_idx in range(len(self.X[0])):
      ids = list(map(lambda i: (i, self.X[i][feature_idx]), indices))
      ids = sorted(ids, key = lambda pair: pair[1])
      ids = np.fromiter(map(lambda pair: pair[0], ids), int)

      left_counters = {}
      right_counters = {}

      for i in ids:
        curr_cls = self.Y[i] - 1
        if not curr_cls in right_counters:
          left_counters[curr_cls] = 0
          right_counters[curr_cls] = 0
        right_counters[curr_cls] += 1

      idx = 0
      while idx < len(ids):
        curr_value = self._calc_value(idx, feature_idx, ids)
        curr_idx = ids[idx]
        while True:
          curr_class = self.Y[curr_idx] - 1
          right_counters[curr_class] -= 1
          left_counters[curr_class] += 1
          idx += 1
          if idx >= len(ids):
            break
          curr_idx = ids[idx]
          if self.X[curr_idx][feature_idx] >= curr_value:
            break

        curr_gini = self._gini(idx, left_counters, len(ids) - idx, right_counters)
        if gini_winner > curr_gini:
          gini_winner = curr_gini
          value_winner = curr_value
          feature_winner = feature_idx

    return Question(feature_winner, value_winner)

  def _entropy(self, group, group_quantity):
    if group_quantity == 0:
      return 0

    entropy = 0
    for cls in range(self.class_quantity):
      if cls in group:
        p = group[cls] / group_quantity
        if (p != 0):
          entropy += p * math.log(p)
    return -entropy

  def _gini(self, left_quantity, left, right_quantity, right):
    entropy_left = self._entropy(left, left_quantity)
    entropy_right = self._entropy(right, right_quantity)
    n = left_quantity + right_quantity
    return left_quantity * entropy_left / n + right_quantity * entropy_right / n

  def _split_on_classes(self, indices, q):
    left, right = np.zeros(2), np.zeros(2)
    for i in indices:
      clazz = 1 if self.Y[i] == 1 else 0
      if self._accept(self.X[i], q.feature_idx, q.value):
        left[clazz] += 1
      else:
        right[clazz] += 1

    def _is_class(yy):
      return 1 if np.argmax(yy) == 1 else -1

    return _is_class(left), _is_class(right)

  def _accept(self, row, feature, value):
    return row[feature] < value

  def _calc_value(self, i, feature, ids):
    return 2e9 if i + 1 == len(ids) else\
      (self.X[ids[i]][feature] + self.X[ids[i + 1]][feature]) / 2
