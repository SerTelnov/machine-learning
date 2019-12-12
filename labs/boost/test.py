import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

import core
import data

TEST_PATH = 'labs/boost/resources/'


def cross_validation(X, Y, boost):
    y_classifier = []
    y_actual = []

    ids = np.arange(len(Y))
    np.random.shuffle(ids)
    ids_batchs = np.array_split(ids, 5)

    for test_num in range(len(ids_batchs)):
        X_train, Y_train = data.train_dataset(X, Y, ids_batchs, test_num)
        boost.fit(X_train, Y_train)

        X_test, Y_test = X[ids_batchs[test_num]], Y[ids_batchs[test_num]]
        for i in range(len(ids_batchs[test_num])):
            y_prediction = boost.classify(X_test[i])
            y_classifier.append(y_prediction)
            y_actual.append(Y_test[i])

    return accuracy_score(y_actual, y_classifier)

X, Y = data.read_data(TEST_PATH + 'geyser.csv')

boost = core.AdaBoost(60)
print('CV result ' + str(cross_validation(X, Y, boost)) + ' for 10 models')