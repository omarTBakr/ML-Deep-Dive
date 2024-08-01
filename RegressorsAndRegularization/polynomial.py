from sklearn import preprocessing
from sklearn import linear_model
from utility import train_test_split_data, calc_RMSE
from matplotlib import pyplot as plt
import numpy as np


def fit_polynomial(model, x, y, degree=1):
    '''
    this function will fit the model to the x after
    applying polynomial with the passed degree
    over it .
    :param model: preferred model
    :param x:  dependent variables
    :param y:  independent variable
    :param degree: degree of the polynomial features
    :return:  modlel , polynomial transformer
    '''
    poly = preprocessing.PolynomialFeatures(degree=degree)
    poly.fit(x)
    model.fit(poly.transform(x), y)
    return model, poly


def test_polynomial(degrees):
    train_errors, test_errors = [], []
    x_train, x_test, y_train, y_test = train_test_split_data()
    model = linear_model.LinearRegression(fit_intercept=True)

    for degree in degrees:
        model, poly = fit_polynomial(model, x_train, y_train, degree)
        train_errors.append(calc_RMSE(model, poly.transform(x_train), y_train))
        test_errors.append(calc_RMSE(model, poly.transform(x_test), y_test))

    return train_errors, test_errors


def test_individual_features(*features, degree=None):
    if degree is None:
        degree = [1, 2, 3]
    x_train, x_test, y_train, y_test = train_test_split_data()
    model = linear_model.LinearRegression(fit_intercept=True)
    train_errors, test_errors = [], []

    for feature in features:
        for degree in degrees:
            selected_train, selelcted_test = x_train[:, feature].reshape(-1, 1), x_test[:, feature].reshape(-1, 1)

            model, poly = fit_polynomial(model, selected_train, y_train, degree)
            train_errors.append(calc_RMSE(model, poly.transform(selected_train), y_train))
            test_errors.append(calc_RMSE(model, poly.transform(selelcted_test), y_test))

    return train_errors, test_errors


if __name__ == '__main__':
    degrees = [1]
    # train,test = test_polynomial(degrees)

    # for degree, train_, test_ in zip(degrees, train, test):
    #     print(f'{degree=}  - {train_=} {test_=}')
    # plt.figure()
    # plt.plot(degrees,train ,label='train')
    # plt.plot( degrees,test , label='test')
    # plt.legend(loc='best')
    # plt.show()

    features = [0, 3, 6]
    train, test = test_individual_features(*features, degree=degrees)

    print(f'{train=}\n,{test=}')
    plt.figure()
    bar_width = 0.25
    plt.bar(features, train, width=bar_width, color='r', label='training')
    plt.bar([feature + bar_width for feature in features], test, width=bar_width, color='b', label='testing')

    plt.xlabel('Features', fontweight='bold', fontsize=15)

    plt.ylabel('Error', fontweight='bold', fontsize=15)

    plt.legend(loc='best')
    plt.show()
