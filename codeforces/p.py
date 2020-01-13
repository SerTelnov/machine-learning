import math

def calc(x):
	avg = sum(x) / len(x)
	total = 0.0
	for val in x:
		total += math.pow(val - avg, 2)
	return total / len(x)

k = int(input())
n = int(input())

xs = [[] for _ in range(k)]

for _ in range(n):
  x, y = (int(val) for val in input().split())
  xs[x - 1].append(y)

result = 0.0

for curr in xs:
  if len(curr) != 0:
    result += (len(curr) / n) * calc(curr)

print(result)
