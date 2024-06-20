import pandas as pd


def _load_data():
    # hard coding file location is a problem .
    try:
        with open('DataProcessing/dataset_200x4_regression.csv') as file:
            df = pd.DataFrame(pd.read_csv(file))
    except FileNotFoundError:
        with open('dataset_200x4_regression.csv') as file:
            df = pd.DataFrame(pd.read_csv(file))
    return df


def load_data():
    df = _load_data()
    # 'Target'
    x_features = ['Feat1', 'Feat2', 'Feat3']
    y_features = ['Target']
    X = df[x_features].to_numpy()
    Y = df[y_features].to_numpy()
    return X, Y


def load_best_feature():
    df = _load_data()
    #  get the best feature  according to corr matrix

    # we are interspersed in the last column
    corr_series = df.corr().iloc[:, -1][:-1]

    # corerelation could be negative also
    corr_series = corr_series.abs()

    best_corr_index = corr_series.argmax()
    # best feature name
    name = corr_series.index[best_corr_index]
    # print(f'{best_corr_index}, { name =}')
    # 'Target'
    x_features = [name]
    y_features = ['Target']
    X = df[x_features].to_numpy()
    Y = df[y_features].to_numpy()
    return X, Y
