import pandas as pd
import numpy as np
from sklearn import preprocessing


# resource_path = "labs\kNN\iris.csv"

def read_data(resource_path):
  df = pd.read_csv(resource_path)

  df['Species'] = df['Species'].apply(lambda name: {"setosa": 0, "versicolor": 1, "virginica":2}[name])

  x = df.drop(['Species'], axis=1).values
  min_max_scaler = preprocessing.MinMaxScaler()
  X = min_max_scaler.fit_transform(x)
  Y = df['Species'].values

  return X, Y
