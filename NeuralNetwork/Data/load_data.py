import pandas as pd
import os
from sklearn import model_selection
from sklearn import impute
from sklearn.preprocessing import  MinMaxScaler



abs_path = os.path.join('Data', 'data2_200x30.csv')
rel_path = 'data2_200x30.csv'


def _load_data():
    try:
        data = pd.read_csv(abs_path)

    except FileNotFoundError:
        data = pd.read_csv(rel_path)

    # data = pd.DataFrame(data)
    return data.iloc[:, :-1].to_numpy(), data.iloc[:, -1].to_numpy().reshape(-1, 1)


def process_data(x, y):
    '''

    :param x: training data (depennt variables)
    :param y: target variable (indpependent variable)
    :return: the data transformed
    transformation done using kNN imputer
    Note : the imputer is fitted to the train then being used
        to transform the val data to prevent any data leakage
    '''
    knnImputer_x = impute.KNNImputer(n_neighbors=10)
    x_train, x_val, y_train, y_val = model_selection.train_test_split(x, y, train_size=0.4)
    knnImputer_x.fit(x_train)

    # filling na values for x
    x_train = knnImputer_x.transform(x_train)
    x_val = knnImputer_x.transform(x_val)

    # filling na values for y
    knnImputer_y = impute.KNNImputer(n_neighbors=10)
    knnImputer_y.fit(y_train)

    y_train, y_val = (knnImputer_y.transform(y_train), knnImputer_y.transform(y_val))
    return (x_train, x_val, y_train, y_val)


def load_data():
    x, y = _load_data()
    return process_data(x, y)


if __name__ == '__main__':
    pass
    # data = load_data()
    # print(x.shape , y.shape)

    # print(data.isnull().sum())
    # data is clean !
