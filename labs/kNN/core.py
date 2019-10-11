import math

def safe_division(a, b):
    if a == 0.0:
        return 0
    elif b == 0.0:
        return 1
    return a / b

# kernels

KERNEL_FUNCTION_NAMES = ['uniform', 'triangular', 'epanechnikov',\
                         'quartic', 'triweight', 'tricube',\
                         'gaussian', 'cosine', 'logistic',\
                         'sigmoid' ]

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

# distances

DIST_FUNCTION_NAMES = ['manhattan', 'euclidean', 'chebyshev']

def get_dist_func(dist_name):
    range_num = 4
    def euclidean(x, y):
        sum = 0
        for i in range(range_num):
            sum += (x[i] - y[i]) ** 2
        return math.sqrt(sum)

    def manhattan(x, y):
        sum = 0
        for i in range(range_num):
            sum += abs(x[i] - y[i])
        return sum

    def chebyshev(x, y):
        sum = -1
        for i in range(range_num):
            sum = max(sum, abs(x[i] - y[i]))
        return sum

    func = {
        'manhattan': manhattan,
        'euclidean': euclidean,
        'chebyshev': chebyshev
    }[dist_name]

    def internal(x1, x2, x3, x4, y1, y2, y3, y4):
        return func(\
            [x1, x2, x3, x4],\
            [y1, y2, y3, y3]\
        )

    return internal


def count_dist_and_sort(X_temp, Y_temp, dist_func, row, curr_index, rows_count):
    dists = list(map(lambda x: (dist_func(x, row)), X_temp))
    dist_pairs = list(zip(dists, [i for i in range(rows_count)]))
    dist_pairs.sort(key=lambda p: p[0])

    X = []
    Y = []
    dist_res = []

    for dist, i in dist_pairs:
        if i != curr_index:
            X.append(X_temp[i])
            Y.append(Y_temp[i])
            dist_res.append(dist)

    return X, Y, dist_res


def calc_window(is_fixed, win_param, dists):
    if is_fixed:
        return win_param
    return dists[win_param]

def init_cm():
    cm = []
    for n in range(3):
        cm.append([0] * 3)
    return cm
