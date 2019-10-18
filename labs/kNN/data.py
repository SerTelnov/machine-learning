import pandas as pd
import numpy as np

# resource_path = "labs\kNN\iris.csv"

def read_data(resource_path):
  df = pd.read_csv(resource_path)

  df = df.assign(Scalar_product = lambda x: np.sqrt(\
      x['Sepal.Length'].pow(2) +\
      x['Sepal.Width'].pow(2) +\
      x['Petal.Length'].pow(2) +\
      x['Petal.Width'].pow(2))\
  )

  df['Sepal.Length'] = df['Sepal.Length'] / df['Scalar_product']
  df['Sepal.Width'] = df['Sepal.Width'] / df['Scalar_product']
  df['Petal.Length'] = df['Petal.Length'] / df['Scalar_product']
  df['Petal.Width'] = df['Petal.Width'] / df['Scalar_product']
  df = df.drop(['Scalar_product'], axis=1)
  df['Species'] = df['Species'].apply(lambda name: {"setosa": 0, "versicolor": 1, "virginica":2}[name])

  X = df.drop(['Species'], axis=1).to_numpy()
  Y = df['Species'].values

  return X, Y
