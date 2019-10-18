import numpy as np
import pandas as pd

import core
import data

FIXED_DIVIDER = 500.0

X, Y = data.read_data("labs/kNN/iris.csv")

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
