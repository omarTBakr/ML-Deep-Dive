import numpy as np


def generate_uniform_data(func):
    # center equation will be 4*x^2 +4X + 6
    X = np.linspace(-5, 5, 100)
    X_new = X.reshape(-1, 1)
    Y = np.array(list(func(value) for value in X_new)).reshape(-1, 1)
    Y = np.add(Y, np.random.randint(-200, 200, 100).reshape(-1, 1))

    return X_new, Y


def get_some_insights():
    X, Y = generate_uniform_data()
    print("deimentions of the data".center(100, '#'))
    print(f'{X.shape=} , {Y.shape=}')
