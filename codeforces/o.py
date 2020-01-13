
k, n = int(input()), int(input())

mapper = [[] for _ in range(k + 1)]
values = [0] * n

in_class = 0
all_class = 0
for i in range(n):
  value, curr_class = list(map(lambda x : int(x), input().split()))

  values[i] = value
  mapper[curr_class].append(value)

values.sort()
for i in range(1, k + 1):
  mapper[i].sort()

for i in range(1, n):
  all_class += i * (n - i) * (values[i] - values[i - 1])

for kk in range(1, k + 1):
  nn = len(mapper[kk])
  for i in range(1, nn):
    in_class += i * (nn - i) * (mapper[kk][i] - mapper[kk][i - 1])

in_class *= 2
all_class *= 2

print(in_class)
print(all_class - in_class)