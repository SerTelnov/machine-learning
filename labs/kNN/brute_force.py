import numpy as np
import pandas as pd

import core


resource_path = "labs\kNN\iris.csv"
df = pd.read_csv(resource_path)

df = df.assign(Scalar_product = lambda x: np.sqrt(\
    x['Sepal.Length'].pow(2) +\
    x['Sepal.Width'].pow(2) +\
    x['Petal.Length'].pow(2) +\
    x['Petal.Width'].pow(2))\
)

df['Sepal.Length'] = df['Sepal.Length'] / df['Scalar_product']
df['Sepal.Width'] = df['Sepal.Width'] / df['Scalar_product']
df['Petal.Length'] = df['Petal.Length'] / df['Scalar_product']
df['Petal.Width'] = df['Petal.Width'] / df['Scalar_product']
df = df.drop(['Scalar_product'], axis=1)
df['Species'] = df['Species'].apply(lambda name: {"setosa": 0, "versicolor": 1, "virginica":2}[name])

X = df.drop(['Species'], axis=1).to_records(index=False)
Y = df['Species'].values

rows_count = len(Y)

class BruteForceWinner(object):
    def __init__(self, dist_func_name, kernel_func_name, window_param, is_fix, f_value):
        self.dist_func_name = dist_func_name
        self.kernel_func_name = kernel_func_name
        self.window_param = window_param
        self.is_fix = is_fix
        self.f_value = f_value


curr_winner = BruteForceWinner("", "", "", "", -1)

for i in range(rows_count):
    curr_row = X[i]
    for dist_name in core.DIST_FUNCTION_NAMES:
        curr_dist_func = core.get_dist_func(dist_name)
        curr_X, curr_Y, curr_dists = core.count_dist_and_sort(X, Y, curr_dist_func, curr_row, i, rows_count)
        for kernel_name in core.KERNEL_FUNCTION_NAMES:
            curr_kernel_func = core.get_kernel_func(kernel_name)
            for k in range(rows_count - 1):
                curr_cm = core.init_cm()
                competition = [] * 3
                win = core.calc_window(False, k, curr_dists)

                for j in range(rows_count):
                    if i != j:
                        value = curr_kernel_func(core.safe_division(curr_dists[j], win))
                        competition[y[j]] += value
