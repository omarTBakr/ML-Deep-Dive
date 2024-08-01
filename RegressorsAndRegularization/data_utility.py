import os
import pandas as pd
import numpy as np

np.random.seed(17)
path = os.path.join('Data', 'data2_200x30.csv')


def get_data_frame():
    df = pd.DataFrame(pd.read_csv(path))
    return df


def get_numpy():
    df = get_data_frame()
    return df.to_numpy()


def get_data_split_numpy():
    data = get_numpy()

    return data[:, :-1], data[:, -1]
