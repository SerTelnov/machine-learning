import numpy as np
import matplotlib.pyplot as plt


class kNNWinner:
    def __init__(self, kernel_name, dist_name, window_option, f_measure):
        self.kernel_name = kernel_name
        self.dist_name = dist_name
        self.window_option = window_option
        self.f_measure = f_measure

    def draw(self, steps):
        plt.plot(f_measure, steps)

        plt.suptitle("kernel: '" + self.kernel_name + "' dist_name: '" + self.dist_name + "'")
        plt.xlabel('f measure')
        plt.ylabel('window steps')

        plt.show()


def read_options(win_option):
    print("tipe fixed options")

    print("kernel_name")
    kernel_name = input()

    print("dist_name")
    dist_name = input()

    print("f_measure")
    f_measure = (float(f) for f in input().split())
    return kNNWinner(kernel_name, dist_name, win_option, f_measure)

fixed_options = read_options("fixed")
variable_options = read_options("variable")

fixed_options.draw()
variable_options.draw()
