import pandas as pd
import numpy as np
from sklearn import preprocessing

def read_data(path):
  df = pd.read_csv(path)
  df['class_numeric'] = df['class'].apply(lambda value: {'P': 1, 'N': -1}[value])

  min_max_scaler = preprocessing.MinMaxScaler()
  X = min_max_scaler.fit_transform(df[['x', 'y']].values)
  Y = df['class_numeric'].to_numpy()

  return X, Y

def train_dataset(X, Y, ids_batchs, test_num):
    train_ids = __merge(ids_batchs, test_num)
    return X[train_ids], Y[train_ids]

def __merge(ids_batchs, test_num):
    ids = np.array([], dtype=np.int64)
    for i in range(len(ids_batchs)):
        if i != test_num:
            ids = np.concatenate((ids, ids_batchs[i]), axis = 0)
    return ids
