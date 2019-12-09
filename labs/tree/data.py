import pandas as pd
import numpy as np


# RESOURCES_PATH = 'labs/tree/resources/'
RESOURCES_PATH = 'resources/'

def read_data(path):
  df = pd.read_csv(RESOURCES_PATH + path)
  return df.drop(['y'], axis=1).to_numpy(), df['y'].to_numpy()
