import math

def safe_division(a, b):
    if a == 0.0:
        return 0
    elif b == 0.0:
        return 1
    return a / b
 
# kernels

kernels_with_conditions = ['uniform', 'triangular', 'epanechnikov', 'quartic',
                            'triweight', 'tricube', 'cosine']

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

# distances

def euclidean(x, y, p):
    sum = 0
    for i in range(p):
        sum += (x[i] - y[i]) ** 2
    return math.sqrt(sum)

def manhattan(x, y, p):
    sum = 0
    for i in range(p):
        sum += abs(x[i] - y[i])
    return sum

def chebyshev(x, y, p):
    sum = -1
    for i in range(p):
        sum = max(sum, abs(x[i] - y[i]))
    return sum

def get_dist_formular(dist_name):
    return {
        'manhattan': manhattan,
        'euclidean': euclidean,
        'chebyshev': chebyshev
    }[dist_name]

def get_kernel_func(kernel_name):
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
    
    if kernel_name in kernels_with_conditions:
        return internal
    return func

n, m = (int(i) for i in input().split())

X_temp = []
Y_temp = [0] * n

for i in range(n):
    arr = [int(v) for v in input().split()]
    X_temp.append(arr[:m])
    Y_temp[i] = arr[m]

request = [int(value) for value in input().split()]

dist_formular = get_dist_formular(input())
kernel_func = get_kernel_func(input())

dists = list(map(lambda x: (dist_formular(x, request, m)), X_temp))

dist_pairs = list(zip(dists, [i for i in range(n)]))
dist_pairs.sort(key=lambda p: p[0])

X = []
Y = []
for dist, i in dist_pairs:
    X.append(X_temp[i])
    Y.append(Y_temp[i])

def calc_window(is_fixed, win_param):
    if is_fixed:
        return win_param
    return dist_pairs[win_param][0]

is_fixed = input() == 'fixed'
h = int(input())

window_param = calc_window(is_fixed, h)

def eval():
    sum1 = 0
    sum2 = 0
    for i in range(n):
        value = kernel_func(safe_division(dist_pairs[i][0], window_param))
        sum1 += Y[i] * value
        sum2 += value
    return sum1, sum2

sum1, sum2 = eval()

if (sum1 == 0.0 and sum2 == 0.0):
    kernel_func = lambda x : x
    sum1, sum2 = eval()

print(safe_division(sum1, sum2))
