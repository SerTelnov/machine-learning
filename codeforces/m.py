import math


def read_data():
  n = int(input())

  pairs = []
  for _ in range(n):
    pairs.append(list(map(lambda x : int(x), input().split())))
  return pairs

def pearson(pairs):
  product_sum = 0.0
  sum1, sum2 = 0, 0
  squares1, squares2 = 0, 0

  for x, y in pairs:
    sum1 += x
    sum2 += y
    product_sum += x * y
    squares1 += x * x
    squares2 += y * y

  size = len(pairs)
  numerator = product_sum - ((sum1 * sum2) / size)
  denominator = math.sqrt(\
    (squares1 - (sum1 * sum1) / size) *\
    (squares2 - (sum2 * sum2) / size)\
  )

  return numerator / denominator if denominator != 0 else 0

pairs = read_data()
print(pearson(pairs))
