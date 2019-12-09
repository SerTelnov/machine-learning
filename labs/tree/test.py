import numpy as np

import core
import data

X_train, Y_train = data.read_data('01_train.csv')
tree = core.TreeBuilder(X_train, Y_train, 20).build_tree()

X_test, Y_test = data.read_data('01_test.csv')
for i in np.arange(len(X_test)):
  answer = tree.classify(X_test[i])
  if answer != Y_test[i]:
    print('error expected ' + str(Y_test[i]) + ' but was ' + str(answer))
