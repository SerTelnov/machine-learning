import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score, f1_score

import core
import data
import kermels

TEST_PATH = 'labs/svm/resources/'

C = 1

# def cross_validation(X, Y, kermel_func):
#     y_classifier = []
#     y_true = []

#     ids = np.arange(len(Y))
#     np.random.shuffle(ids)
#     ids_batchs = np.array_split(ids, 5)

#     for test_num in range(len(ids_batchs)):
#         X_train, Y_train = data.train_dataset(X, Y, ids_batchs, test_num)
#         classifier = core.SVM(X_train, Y_train, kermel_func).evaluate(C)

#         for i in ids_batchs[test_num]:
#             y_prediction = classifier.classify(X[i])

#             y_classifier.append(y_prediction)
#             y_true.append(Y[i])

#     return f1_score(y_true, y_classifier)

# # linear kernel

# X, Y = data.read_data(TEST_PATH + 'geyser.csv')
# f = -1
# for c in np.linspace(0,1,11):
#     print("evaluating for C = " + str(c))
#     curr_f = cross_validation(X, Y, kermels.linear_kernel(c))
#     if (f < curr_f):
#         f = curr_f
#         best_const = c

# print("best const = " + str(best_const))
# print("cross validation result = " + str(f))

def plot_decision(x, y, classifier, title, idx=[], resolution=0.02):
    plt.rcParams['figure.figsize'] = [10, 7]
    colors = ('blue', 'green')
    cmap = ListedColormap(colors[:len(np.unique(y))])        

    # x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    # x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1

    # xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    # classes = classifier.classify(np.c_[xx1.ravel(), xx2.ravel()])
    classes = classifier.classify(x)
    # classes = classes.reshape(xx1.shape)
    # classes = classifier.classify(x)

    plt.contourf(x, classes, alpha=0.4, cmap=cmap)

    for elem_x, elem_y in zip(x, y):
        if elem_y == -1:
            plt.scatter(elem_x[0], elem_x[1], s = 40, color=colors[0])
        else:
            plt.scatter(elem_x[0], elem_x[1], s = 40, color=colors[1])

    plt.title(title)
    plt.show()

for path in ['chips', 'geyser']:
    X, Y = data.read_data(TEST_PATH + path + '.csv')
    ids_batches = data.split_indices_data(len(Y))
    X_train, Y_train = data.train_dataset(X, Y, ids_batches, 0)

    classifier = core.SVM(X_train, Y_train, kermels.linear_kernel(1.0)).evaluate(C)
    plot_decision(X, Y, classifier, path + ", linear kernel")

    classifier = core.SVM(X_train, Y_train, kermels.polynomial_kernel(2)).evaluate(C)
    plot_decision(X, Y, classifier, path + ", polynomial kernel")

    classifier = core.SVM(X_train, Y_train, kermels.gaussian_kernel(2)).evaluate(C)
    plot_decision(X, Y, classifier, path + ", gaussian kernel")