import numpy as np
import matplotlib.pyplot as plt

import core
import data


class kNNWinner:
    def __init__(self, kernel_name, dist_name, is_fixed):
        self.kernel_name = kernel_name
        self.dist_name = dist_name
        self.is_fixed = is_fixed

    def draw(self, X, Y):
        all_dists, max_dist = core.count_dists(X, core.get_dist_func(self.dist_name))
        win_step = max_dist / 500

        steps = np.arange(0, 0.5, win_step) if self.is_fixed else range(1, 100)
        f_measure = []

        for curr_win in steps:
            f = core.brute_force_params(curr_win, all_dists, core.get_kernel_func(self.kernel_name), self.is_fixed, Y)
            f_measure.append(f)

        plt.plot(steps, f_measure)

        plt.suptitle("kernel: '" + self.kernel_name + "' dist_name: '" + self.dist_name + "'")
        plt.ylabel('f measure')
        plt.xlabel('window steps')

        plt.show()


def read_options(win_option):
    print("kernel_name")
    kernel_name = input()

    print("dist_name")
    dist_name = input()

    return kNNWinner(kernel_name, dist_name, win_option)


X, Y = data.read_data("labs/kNN/iris.csv")

print("type option (fixed/variable)")
is_fixed = input() == "fixed"

option = read_options(is_fixed)
option.draw(X, Y)
