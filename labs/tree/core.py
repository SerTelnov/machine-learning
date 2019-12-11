import numpy as np

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

  def classify(self, row):
    if self._accept(row):
      return self.left.classify(row)
    else:
      return self.right.classify(row)

  def _accept(self, row):
    return row[self.question.feature_name] < self.question.feature_value

class Leaf(Tree):
  def __init__(self, class_name):
    self.class_name = class_name

  def classify(self, row):
    return self.class_name

class TreeBuilder:
  def __init__(self, X, Y, max_depth):
    self.X = X
    self.Y = Y
    self.max_depth = max_depth
    self.class_quantity = len(self._collect_classes(np.arange(len(Y))))

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
    gini_winner = -2e9

    for feature_idx in range(len(self.X[0])):
      indices = list(enumerate(map(lambda i: self.X[i][feature_idx], indices)))
      indices = sorted(indices, key = lambda pair: pair[1])
      indices = np.fromiter(map(lambda pair: pair[0], indices), int)

      left_counters = np.zeros(self.class_quantity)
      right_counters = np.zeros(self.class_quantity)

      for i in indices:
        right_counters[self.Y[i] - 1] += 1

      left_score, right_score = 0.0, 0.0

      for x in right_counters:
        right_score += x * x

      value = -2e9
      gini = right_score / len(indices)

      for idx, i in enumerate(indices):
        curr_class = self.Y[i] - 1
        left_item_count = idx + 1
        right_item_count = len(indices) - left_item_count

        x = right_counters[curr_class]
        xx = right_counters[curr_class] - 1
        right_score += xx * xx - x * x

        x = left_counters[curr_class]
        xx = left_counters[curr_class] + 1
        left_score += xx * xx - x * x

        right_counters[curr_class] -= 1
        left_counters[curr_class] += 1

        curr_gini = left_score / left_item_count
        curr_gini += right_score / right_item_count if right_item_count != 0 else 0

        if gini < curr_gini:
          gini = curr_gini
          value = self._calc_value(i, feature_idx, indices)

      if gini_winner < gini:
        gini_winner = gini
        value_winner = value
        feature_winner = feature_idx

    left, right = self._split(indices, value_winner, feature_winner)
    q = Question(feature_winner, value_winner)
    return left, right, q

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
    return Leaf(np.argmax(classes))

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
