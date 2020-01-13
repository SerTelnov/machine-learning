import math

k1, k2 = (int(val) for val in input().split())
n = int(input())

k1Count = [0] * k1
k2Count = [0] * k2

O = [{} for _ in range(k1)]

for _ in range(n):
  x1, x2 = (int(val) - 1 for val in input().split())
  k1Count[x1] += 1
  k2Count[x2] += 1

  O[x1][x2] = (O[x1].get(x2) or 0) + 1

k1Sum = sum(k1Count)
k2Sum = sum(k2Count)

totalSum = float(k1Sum * k2Sum) / n
for i in range(k1):
  for key, value in O[i].items():
    fe = float(k1Count[i] * k2Count[key]) / n
    totalSum += math.pow(value - fe, 2) / fe - fe

print("{:.16f}".format(totalSum))
