from sklearn import preprocessing
from sklearn import linear_model
from utility import train_test_split_data, calc_RMSE
from matplotlib import pyplot as plt
import numpy as np


def monomail_features(degree, x: np.array):
    transformed = x
    for degree in range(2, degree + 1):
        transformed = np.hstack((transformed, np.power(x, degree)))

    return transformed


def test_monomial(degrees):
    x_train, x_test, y_train, y_test = train_test_split_data()
    train_errors, test_errors = [], []
    model = linear_model.LinearRegression(fit_intercept=True)
    for degree in degrees:
        x_train_transformed = monomail_features(degree, x_train)
        x_test_transformed = monomail_features(degree, x_test)
        model.fit(x_train_transformed, y_train)

        train_errors.append(calc_RMSE(model, x_train_transformed, y_test))

        test_errors.append(calc_RMSE(model, x_test_transformed, y_test))

    return train_errors, test_errors


if __name__ == "__main__":
    degrees = [1, 2, 3, 4]
    train, test = test_monomial(degrees)
    for degree, train_, test_ in zip(degrees, train, test):
        print(f'{degree=}  - {train_=} {test_=}')

    plt.plot(degrees, train, label='train')
    plt.plot(degrees, test, label='test')
    plt.legend(loc='best')
    plt.show()
