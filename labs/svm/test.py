import numpy as np

import core
import data
import kermels

TEST_PATH = 'labs/svm/resources/'
# TEST_PATH = 'resources/'

C = 1

def classifier(x, w, b):
    value = np.sum(x * w) + b
    return 1 if value > 0 else -1

def cross_validation(X, Y, kermel_func):
    ids = np.arange(Y.size)
    np.random.shuffle(ids)
    ids_batchs = np.array_split(ids, 5)
    contingency_matrix = [[0] * 2 for _ in range(2)]

    for test_num in range(len(ids_batchs)):
      print("test number #" + str(test_num))
      X_train, Y_train = data.train_dataset(X, Y, ids_batchs, test_num)
      w, b = core.SVM(X_train, Y_train, kermel_func).evaluate(C)

      X_test, Y_test = X[ids_batchs[test_num]], Y[ids_batchs[test_num]]
      for i in range(len(ids_batchs[test_num])):
          y_prediction = classifier(X_test[i], w, b)
          
          prediciton_index = 0 if y_prediction == -1 else 1
          actual_index = 0 if Y_test[i] == -1 else 1

          contingency_matrix[prediciton_index][actual_index] += 1

    f, _ = core.eval_f(contingency_matrix)
    return f

# linear kernel

X, Y = data.read_data(TEST_PATH + 'chips.csv')
f = -1
for c in np.linspace(0,1,11):
    print("evaluating for C = " + str(c))
    curr_f = cross_validation(X, Y, kermels.linear_kernel(c))
    if (f < curr_f):
        f = curr_f
        best_const = c

print("best const = " + str(best_const))
print("cross validation result = " + str(f))