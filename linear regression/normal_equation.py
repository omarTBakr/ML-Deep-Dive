import numpy as np
from DataProcessing.preprocessing import minmax_scaler, standard_scaler
from DataProcessing.load_data import load_data
from tqdm import tqdm
from sklearn.metrics import mean_squared_error


def normal_equation(X, Y):
    # padding X
    X_new = np.hstack((np.ones((X.shape[0], 1)), X))
    # solving A x = b
    # A should have a full rank
    A, b = X_new.T @ X_new, X_new.T @ Y

    return np.linalg.solve(A, b)


def error(X, Y, theta):
    X_new = np.hstack((np.ones((X.shape[0], 1)), X))
    N = X_new.shape[0]
    return mean_squared_error(X_new @ theta, Y) / N / 2


if __name__ == '__main__':
    scalers = (minmax_scaler, standard_scaler)
    X, Y = load_data()
    for scaler in tqdm(scalers):
        X, Y = scaler(X, Y)
        print(f'data scalled with {scaler.__name__}'.center(100, '*'))
        theta = normal_equation(X, Y)
        print(f'with {theta =}')
        print(f'with error {error(X , Y , theta)}')

    print('without scaling'.center(100, '*'))
    theta = normal_equation(X, Y)
    print(f'with {theta =}')
    print(f'with error {error(X, Y, theta)}')
