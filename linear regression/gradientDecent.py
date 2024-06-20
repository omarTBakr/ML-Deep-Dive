import numpy as np
from numpy.linalg import norm
from cost import cost_der, cost


def gradient_descent_linear_regression(X, Y, W=None, step_size=0.01, precision=0.0001, max_iter=1000):
    # initializing weight to be random
    if W is None:
        initial_vect = np.random.rand((X.shape[1] + 1)).reshape(-1, 1)
    else:
        initial_vect = W

    cur_vect = np.array(initial_vect)
    last_vect = np.array([float('inf')] * len(cur_vect)).reshape(-1, 1)

    error = [cost(X, Y, initial_vect).item()]
    _iter = 0

    while norm(cur_vect - last_vect) > precision and _iter < max_iter:
        # print(cur_vect)
        last_vect = cur_vect.copy()  # must copy

        gradient = cost_der(X, Y, cur_vect)
        cur_vect = np.subtract(cur_vect, gradient * step_size, casting='unsafe')  # move in opposite direction

        error.append(cost(X, Y, cur_vect).item())
        _iter += 1

    # print(f'The minimum z exists at (x,y) = {cur_vect}')

    return error, cur_vect, _iter
