n, m = (int(i) for i in input().split())

X = []
Y = [0] * n

for i in range(n):
    arr = [int(v) for v in input().split()]
    X.append(arr[:m])
    Y[i] = arr[m]

