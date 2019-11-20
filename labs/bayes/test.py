import numpy as np

import core
import data

def cross_validation(alfa, batch_data):
    contingency_matrix = [[0] * 2 for _ in range(2)]

    for test_idx in range(len(batch_data)):
        train_data = data.concate_except_idx(batch_data, test_idx)
        preparetion = core.train(train_data, [1, 1])

        for test_mail in batch_data[test_idx]:
            is_spam = core.classifier(preparetion, alfa, test_mail)
            contingency_matrix[is_spam][test_mail.is_spam] += 1

    f, _ = core.eval_f(contingency_matrix)
    return f

batch_data = data.read_batch_data()
winner_f = -1

for alfa in np.arange(0.0, 1.0, 0.01):
    print("eval for alfa " + str(alfa))
    f = cross_validation(alfa, batch_data)
    if winner_f < f:
        winner_f = f
        winner_alfa = alfa

print("Winner alfa " + str(alfa) + " with cross validation result " + str(winner_f))

# def test(batch_data, alfa, lambda_spam, lambda_legit):
#     for test_idx in range(len(batch_data)):
#         train_data = data.concate_except_idx(batch_data, test_idx)
#         preparetion = core.train(train_data, [lambda_legit, lambda_spam])

#         for test_mail in batch_data[test_idx]:
#             is_spam = core.classifier(preparetion, alfa, test_mail)
#             if is_spam == 1 and test_mail.is_spam == 0:
#                 return False
#         return True

# alfa = 0.99

# for lambda_spam in range(1, 1000, 10):
#     for lambda_legit in range(lambda_spam, 1000, 10):
#         if test(batch_data, alfa, lambda_spam, lambda_legit):
#             print("No mistakes at lambda spam " + str(lambda_spam) + " legit " + str(lambda_legit))