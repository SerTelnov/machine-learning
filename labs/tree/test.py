import numpy as np
from sklearn.metrics import accuracy_score

import data
import core

HEIGHT_UNLIMIT = 1e10


test = 0

test_str = str(test).zfill(2)
print('start for test #' + str(test))

accuracy_winner = -1
h_winner = -1

X_train, Y_train = data.read_data(test_str + '_train.csv')
X_test, Y_test = data.read_data(test_str + '_test.csv')

tree_builder = core.TreeBuilder(X_train, Y_train, HEIGHT_UNLIMIT)
tree = tree_builder.build_tree()
print('build tree with height ' + str(tree.height))

for _ in range(1, tree.height):
    h = tree.height - 1
    # tree = tree_builder.reduce_height(tree, h)
    predictions = list(map(lambda x : tree.classify(x), X_test))
    accuracy = accuracy_score(Y_test, predictions)

    print('current h = ' + str(h) + ' with accuracy ' + str(accuracy))
    if accuracy_winner < accuracy:
        accuracy_winner = accuracy
        h_winner = h

print('winner h = ' + str(h_winner) + ' with accuracy ' + str(accuracy_winner))

forest = core.RandomForest(X_train, Y_train)

predictions = list(map(lambda x : forest.classify(x), X_test))
accuracy = accuracy_score(Y_test, predictions)

print('  build random forest with accuracy ' + str(accuracy))
print('  accuracy profite ' + str(accuracy_winner - accuracy))