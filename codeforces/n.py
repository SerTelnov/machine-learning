import math

def read_data():
  n = int(input())

  X, Y = [0] * n, [0] * n
  for i in range(n):
    X[i], Y[i] = list(map(lambda x : int(x), input().split()))
  return X, Y

def rankify(values):
  xx = list(map(lambda i: (i, values[i]), range(len(values))))
  xx = sorted(xx, key = lambda pair: pair[1])
  ranks = [0] * len(values)

  equals_count = 0
  ranks[xx[0][0]] = 1

  for i in range(1, len(values)):
    r, s = 1, 1
    if xx[i][1] == xx[i - 1][1]:
      equals_count += 1
    else:
      equals_count = 0

    s = equals_count + 1
    r = i - equals_count + 1

    ranks[xx[i][0]] = r + (s - 1) * 0.5

  return ranks

def correlation_сoefficient(X, Y):
  sum_X, sum_Y, sum_XY = 0, 0, 0
  squareSum_X, squareSum_Y = 0.0, 0.0

  for i in range(len(X)):
    sum_X = sum_X + X[i]
    sum_Y = sum_Y + Y[i]
    sum_XY = sum_XY + X[i] * Y[i]

    squareSum_X = squareSum_X + X[i] * X[i]
    squareSum_Y = squareSum_Y + Y[i] * Y[i]

  n = len(X)
  return (n * sum_XY -\
                sum_X * sum_Y) /\
                math.sqrt((n * squareSum_X -\
                      sum_X * sum_X) *\
                      (n * squareSum_Y -\
                      sum_Y * sum_Y))

X, Y = read_data()
print(correlation_сoefficient(rankify(X), rankify(Y)))
