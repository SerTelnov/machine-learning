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

# distances

DIST_FUNCTION_NAMES = ['manhattan', 'euclidean', 'chebyshev']

def count_dists(X, dist_func):
    all_dists = [[Distance(-1, -1)] * len(X) for _ in range(len(X))]
    max_dist = - 1

    for i in range(len(X)):
        for j in range(len(X)):
            curr_dist = dist_func(X[i], X[j])
            all_dists[i][j] = Distance(curr_dist, j)
            max_dist = max(curr_dist, max_dist)
        all_dists[i].sort(key=lambda dist: dist.value)

    return all_dists, max_dist


class Distance:
    def __init__(self, value, index):
        self.value = value
        self.index = index

def calc_f(pr, re):
    if (pr + re == 0):
        return 0
    return 2 * (pr * re) / (pr + re)

def eval_f(rows):
    k = len(rows)
    tp = [0] * k
    fp = [0] * k
    fn = [0] * k
    weight = [0] * k

    sum = 0
    for i, cur_row in enumerate(rows):
        for idx, val in enumerate(cur_row):
            sum += val
            if (idx != i):
                fn[idx] += val
                fp[i] += val
            else:
                tp[i] += val

    if sum != 0:
        for i in range(k):
            weight[i] = (tp[i] + fp[i]) / sum

    pr = [0] * k
    re = [0] * k

    for i in range(k):
        if tp[i] + fp[i] != 0:
            pr[i] = tp[i] / (tp[i] + fp[i])
        if tp[i] + fn[i] != 0:
            re[i] = tp[i] / (tp[i] + fn[i])

    pr_macro = 0.0
    re_macro = 0.0

    for i in range(k):
        pr_macro += weight[i] * pr[i]
        re_macro += weight[i] * re[i]


    f_micro = 0

    for i in range(k):
        f_micro += weight[i] * calc_f(pr[i], re[i])

    return calc_f(pr_macro, re_macro), f_micro


def brute_force_params(win_param, all_dists, kernel_func, is_fixed, Y):
    contingency_matrix = [[0] * 3 for _ in range(3)]
    for curr_obj_idx in range(len(Y)):
        window = win_param if is_fixed else all_dists[curr_obj_idx][win_param].value

        competition = [0] * 3
        for dist_obj in all_dists[curr_obj_idx][1:]:
            curr_y = Y[dist_obj.index]
            competition[curr_y] += kernel_func(safe_division(dist_obj.value, window))

        winner_y, _ = max(enumerate(competition), key=lambda pair: pair[1])
        contingency_matrix[winner_y][Y[curr_obj_idx]] += 1
    f, _ = eval_f(contingency_matrix)
    return f

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
