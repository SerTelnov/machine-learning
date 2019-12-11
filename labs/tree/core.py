import numpy as np
import math

class Question:
  def __init__(self, feature_name, feature_value):
    self.feature_name = feature_name
    self.feature_value = feature_value

class Tree:
  def __init__(self):
    super().__init__()

  def classify(self, row):
    pass

class Node(Tree):
  def __init__(self, left, right, question):
    self.left = left
    self.right = right
    self.question = question
    self.height = max(left.height, right.height) + 1

  def classify(self, row):
    if self._accept(row):
      return self.left.classify(row)
    else:
      return self.right.classify(row)

  def _accept(self, row):
    return row[self.question.feature_name] < self.question.feature_value

class Leaf(Tree):
  def __init__(self, class_name, indices):
    self.class_name = class_name
    self.indices = indices
    self.height = 0

  def classify(self, row):
    return self.class_name

class TreeBuilder:
  def __init__(self, X, Y, max_depth):
    self.X = X
    self.Y = Y
    self.max_depth = max_depth
    self.class_quantity = len(self._collect_classes(np.arange(len(Y))))

  def reduce_height(self, tree, new_height):
    return self.reduce_tree(tree, 0, new_height)

  def reduce_tree(self, tree, curr_depth, limit):
    if isinstance(tree, Node):
      left = self.reduce_tree(tree.left, curr_depth + 1, limit)
      right = self.reduce_tree(tree.right, curr_depth + 1, limit)
      if curr_depth < limit:
        return Node(left, right, tree.question)
      else:
        return self.union(left, right)
    elif isinstance(tree, Leaf):
      return tree
    raise Exception('Invalid pattern')

  def union(self, a, b):
    indicies = a.indices + b.indices
    return self._terminal(indicies)

  def build_tree(self):
    return self._build_tree(np.arange(len(self.Y)), 0)

  def _build_tree(self, indices, curr_depth):
    if curr_depth >= self.max_depth or self._is_one_class(indices):
      return self._terminal(indices)

    left, right, q = self._make_split(indices)
    if len(left) == 0 or len(right) == 0:
      return self._terminal(left + right)

    left_tree = self._build_tree(left, curr_depth + 1)
    right_tree = self._build_tree(right, curr_depth + 1)

    return Node(left_tree, right_tree, q)

  def _make_split(self, indices):
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

    left, right = self._split(indices, value_winner, feature_winner)
    q = Question(feature_winner, value_winner)
    return left, right, q

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

  def _split(self, indices, value, feature):
    left, right = [], []
    for i in indices:
      if self._accept(self.X[i], feature, value):
        left.append(i)
      else:
        right.append(i)
    return left, right

  def _accept(self, row, feature, value):
    return row[feature] < value

  def _calc_value(self, i, feature, ids):
    return 2e9 if i + 1 == len(ids) else\
      (self.X[ids[i]][feature] + self.X[ids[i + 1]][feature]) / 2

  def _terminal(self, indices):
    classes = np.zeros(self.class_quantity)
    for i in indices:
      classes[self.Y[i] - 1] += 1
    return Leaf(np.argmax(classes) + 1, indices)

  def _is_one_class(self, ids):
    return len(self._collect_classes(ids)) == 1

  def _collect_classes(self, ids):
    classes = {}
    for i in ids:
      id = self.Y[i]
      if not id in classes:
        classes[id] = 0
      classes[id] += 1
    return classes
