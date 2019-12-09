import pandas as pd
import numpy as np


RESOURCES_PATH = 'labs/tree/resources/'

def read_data(path):
  df = pd.read_csv(RESOURCES_PATH + path)
  Y = df['class_numeric'].to_numpy()

  return df.drop(['Y'], axis=1).to_numpy(), df['Y'].to_numpy()


X, Y = read_data('01_test.csv')
