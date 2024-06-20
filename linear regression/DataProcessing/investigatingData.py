import pandas as pd

from load_data import _load_data
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import minmax_scale, StandardScaler


def some_data_insights(df, label='no transformation'):  # data frame object containing the data
    print('investigating data'.center(100, '*'))
    plt.figure(label)
    sns.pairplot(df)

    # lets see some info about the data
    print("head of the data".center(100, '#'))
    print(df.head())

    print('correlation matrix'.center(100, '#'))
    plt.figure(label)
    corr_matrix = df.corr()
    print(corr_matrix)

    print("columns in the data".center(100, '#'))
    print(df.columns)

    print("statistical information about the data".center(100, '#'))
    print(df.describe())

    print('more infor about each column'.center(100, '#'))
    print(df.info())

    plt.figure(label)
    # visualizing correlation matrix
    sns.heatmap(corr_matrix)
    plt.show()


if __name__ == '__main__':
    df = _load_data()
    columns = df.columns
    # print('data row , no preprocessing'.center(100, '*'))
    # some_data_insights(df)
    #
    #
    # # preprocess the data and investigate again
    # print(' data transformed using min max scaler '.center(100, '*'))
    # np_array = df.to_numpy()
    # np_array_transformed = minmax_scale(np_array)
    # df_transformed_minMax = pd.DataFrame( np_array_transformed , columns= columns)
    # some_data_insights(df_transformed_minMax , 'transformed using min max scaler')
    #
    # print(' data transformed using standard scaler  '.center(100, '*'))
    # np_array = df.to_numpy()
    # scaler = StandardScaler()
    # np_array_transformed = scaler.fit_transform(np_array)
    # df_transformed_minMax = pd.DataFrame(np_array_transformed, columns=columns)
    # some_data_insights(df_transformed_minMax, 'transformed using standard scaler')
