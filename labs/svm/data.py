import pandas as pd
import numpy as np
from sklearn import preprocessing

def read_data(path):
  df = pd.read_csv(path)
  df['class_numeric'] = df['class'].apply(lambda value: {'P': 1, 'N': -1}[value])

  min_max_scaler = preprocessing.MinMaxScaler()
  X = min_max_scaler.fit_transform(df[['x', 'y']].values)
  Y = df['class_numeric'].values

  return X, Y
