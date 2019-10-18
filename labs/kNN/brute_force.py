import numpy as np
import pandas as pd

import core

FIXED_DIVIDER = 500.0
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

X = df.drop(['Species'], axis=1).to_numpy()
Y = df['Species'].values

rows_count = len(Y)

print("Choose windows param (fixed/variable)")
is_fixed = input() == "fixed"

f_max = -1
kernel_func_winner = ""
dist_func_winner = ""
window_winner = ""

for dist_name in core.DIST_FUNCTION_NAMES:
    print("### Start count for distance function '" + dist_name + "'")
    curr_dist_func = core.get_dist_func(dist_name)
    all_dists, max_dist = core.count_dists(X, curr_dist_func)
    win_step = max_dist / FIXED_DIVIDER
    for kernel_name in core.KERNEL_FUNCTION_NAMES:
        print("###### Start count for kernel function '" + kernel_name + "'")
        curr_kernel_func = core.get_kernel_func(kernel_name)
        window_range = np.arange(win_step, max_dist, win_step) if is_fixed else range(1, len(X) - 1)

        for curr_window_param in window_range:
            f = core.brute_force_params(curr_window_param, all_dists, curr_kernel_func, is_fixed, Y)
            if f_max < f:
                f_max = f
                kernel_func_winner = kernel_name
                dist_func_winner = dist_name
                window_winner = curr_window_param


print("winners:")
print("kernel function: '" + kernel_func_winner + "'")
print("distance function: '" + dist_func_winner + "'")
print("window: '" + str(window_winner) + "'")
print("winner f measure: '" + str(f_max) + "'")
