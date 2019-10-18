import math

def safe_division(a, b):
    if a == 0.0:
        return 0
    elif b == 0.0:
        return 1
    return a / b 

# distances

def get_dist_func(dist_name):
    def euclidean(x, y):
        sum = 0
        for i in range(len(y)):
            sum += (x[i] - y[i]) ** 2
        return math.sqrt(sum)

    def manhattan(x, y):
        sum = 0
        for i in range(len(y)):
            sum += abs(x[i] - y[i])
        return sum

    def chebyshev(x, y):
        sum = -1
        for i in range(len(y)):
            sum = max(sum, abs(x[i] - y[i]))
        return sum

    return {
        'manhattan': manhattan,
        'euclidean': euclidean,
        'chebyshev': chebyshev
    }[dist_name]

# kernels

def get_kernel_func(kernel_name):
    def uniform(u):
        return 0.5

    def triangular(u):
        return 1 - abs(u)

    def epanechnikov(u):
        return (1 - u ** 2) * 3 / 4

    def quartic(u):
        return ((1 - u ** 2) ** 2) * 15 / 16

    def triweight(u):
        return ((1 - u ** 2) ** 3) * 35 / 32

    def tricube(u):
        return ((1 - abs(u ** 3)) ** 3) * 70 / 81

    def gaussian(u):
        return (math.e ** (-0.5 * (u ** 2))) / math.sqrt(2 * math.pi)

    def cosine(u):
        return (math.pi / 4) * math.cos((math.pi / 2) * u)

    def logistic(u):
        return 1 / ((math.e ** u) + 2 + (math.e ** (-u)))

    def sigmoid(u):
        return (2 / math.pi) * (1 / ((math.e ** u) + (math.e ** (-u))))

    func = {
        'uniform': uniform,
        'triangular': triangular, 
        'epanechnikov': epanechnikov, 
        'quartic': quartic,
        'triweight': triweight,
        'tricube': tricube,
        'gaussian': gaussian,
        'cosine':cosine,
        'logistic': logistic,
        'sigmoid': sigmoid
    }[kernel_name]

    def internal(u):
        if abs(u) >= 1.0:
            return 0
        return func(u)

    if kernel_name in ['uniform', 'triangular', 'epanechnikov', \
                       'quartic', 'triweight', 'tricube', \
                       'cosine']:
        return internal
    return func

def eval(Y, dists, window_param, kernel_func):
    sum1 = 0
    sum2 = 0
    for i in range(n):
        value = kernel_func(safe_division(dists[i], window_param))
        sum1 += Y[i] * value
        sum2 += value
    return sum1, sum2


n, m = (int(i) for i in input().split())

X_temp = []
Y_temp = [0] * n

for i in range(n):
    arr = [int(v) for v in input().split()]
    X_temp.append(arr[:m])
    Y_temp[i] = arr[m]

request = [int(value) for value in input().split()]

dist_func = get_dist_func(input())
kernel_func = get_kernel_func(input())

dists = list(map(lambda x: (dist_func(x, request)), X_temp))

dist_pairs = list(zip(dists, [i for i in range(n)]))
dist_pairs.sort(key=lambda p: p[0])

X = []
Y = []
for dist, i in dist_pairs:
    X.append(X_temp[i])
    Y.append(Y_temp[i])

sorted_dists, _ = zip(*dist_pairs)

is_fixed = input() == 'fixed'
h = int(input())

window_param = h if is_fixed else sorted_dists[h]

sum1, sum2 = eval(Y, sorted_dists, window_param, kernel_func)

if (sum1 == 0.0 and sum2 == 0.0):
    kernel_func = lambda x : x
    sum1, sum2 = eval(Y, sorted_dists, window_param, kernel_func)

print(safe_division(sum1, sum2))
