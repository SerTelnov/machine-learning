import math

kx, _ = (int(val) for val in input().split())
n = int(input())

xs = [{} for _ in range(kx)]
xLens = [0] * kx

for _ in range(n):
  x, y = (int(val) - 1 for val in input().split())
  xLens[x] += 1
  xs[x][y] = (xs[x].get(y) or 0) + 1

totalSum = 0.0
for i in range(kx):
  xLen = xLens[i]
  sumPi = float(xLen) / n
  currSum = 0.0
  for x in xs[i].values():
    if x != 0:
      tmp = x / xLen
      currSum -= tmp * math.log(tmp)
  totalSum += currSum * sumPi

print(totalSum)
