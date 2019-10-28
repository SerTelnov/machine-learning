import random


def read_data():
    n, m = (int(i) for i in input().split())

    X = []
    Y = [0] * n

    for i in range(n):
        arr = [int(v) for v in input().split()]
        curr_x = arr[:m]
        curr_x.append(1)
        X.append(curr_x)
        Y[i] = arr[m]

    return X, Y

def get_mini_batches(X, Y, batch_size):
    batch_size = min(len(Y), batch_size)
    random_idxs = [i for i in range(len(Y))]
    random.shuffle(random_idxs)

    batchs = []

    for i in range(0, len(Y) - 1, batch_size):
        Xs = []
        Ys = []
        count = min(batch_size, len(Y) - i)
        for j in range(count):
            if i + j >= len(Y):
                break
            index = random_idxs[i + j]
            Xs.append(X[index])
            Ys.append(Y[index])
        batchs.append((Xs, Ys))

    return batchs

def update_weight(X, Y, W):
    A = [0] * len(Y)
    for i in range(len(Y)):
        for j in range(len(X[i])):
            A[i] += X[i][j] * W[j]

    Gr = [0] * len(W)
    h = 0.0
    for i in range(len(Y)):
        value = A[i] - Y[i]
        gr_value = value * 2
        dx = 0.0
        for j in range(len(X[i])):
            j_gr_value = X[i][j] * gr_value
            Gr[j] += j_gr_value
            dx += X[i][j] * j_gr_value
        if dx != 0:
            h += value / dx

    if h == 0:
        return W

    h /= len(Y)
    for i in range(len(W)):
        W[i] = W[i] - h * Gr[i] / len(Y)
    return W

def SGD(X, Y):
    w_init = 1 / (2 * len(X[0]))
    W = [random.uniform(-w_init, w_init) for _ in range(len(X[0]))]

    for (X_mini, Y_mini) in get_mini_batches(X, Y, 30):
        W = update_weight(X_mini, Y_mini, W)
    return W

X, Y = read_data()
W = SGD(X, Y)

for w in W:
    print(w)
