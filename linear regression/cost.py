'''
    this module will contain implementation
    - cost function
    -  partial derivatives of the cost function

    preconditions are
    ### this is critical ###
    data (X , Y , W ) have two dimensions not a single One


'''

import numpy as np
from scipy.optimize import check_grad


def error(X: np.array, Y: np.array, W: np.array):
    '''return X_new @ W - Y
    dimentions : column vector is the same as Y
    '''

    X_new = np.hstack((np.ones((X.shape[0], 1)), X))
    assert X_new.shape[1] == W.shape[0], 'inner dimension must be the same'

    return (X_new @ W) - Y


def cost(X: np.array, Y: np.array, W: np.array):
    # add new column of ones to X
    # X_new =np.hstack((np.ones(( X.shape[0] , 1)) , X) )
    N = X.shape[0]
    # return np.sum(
    #     np.square(error(X ,Y,W))
    #         )/N/2
    _error = error(X, Y, W)
    return _error.T @ _error / N / 2


def cost_der(X: np.array, Y: np.array, W: np.array):
    X_new = np.hstack((np.ones((X.shape[0], 1)), X))
    N = X.shape[0]
    return X_new.T @ error(X, Y, W) / N


def pseudo_gradient_check(cost, der_cost, X, Y, W, epsilon=1.49e-8):
    numerical_gradient = np.zeros_like(W)
    for index in range(W.shape[0]):
        w_plus, w_minus = np.copy(W), np.copy(W)
        w_plus[index] += epsilon
        w_minus[index] -= epsilon

        right_lim, left_lim = cost(X, Y, w_plus).item(), cost(X, Y, w_minus).item()
        numerical_gradient[index] = (right_lim - left_lim) / (2 * epsilon)
    return numerical_gradient


if __name__ == '__main__':
    # # let's test
    W = np.array([1, 1], dtype=np.float64).reshape(-1, 1)
    Y = np.array([5.0, 5.2, 5.4, 5.8, 6.0]).reshape(-1, 1)
    X = np.array([0, 0.2, 0.4, 0.8, 1.0]).reshape(-1, 1)

    print(cost(X, Y, W))
    analatical_gradient = cost_der(X, Y, W)

    print('checking numerical gradient'.center(100, '*'))
    numerical_gradient = pseudo_gradient_check(cost, cost_der, X, Y, W)

    print(f'pseudo_gradient_check', np.linalg.norm(np.fabs(numerical_gradient - analatical_gradient)))

    # Wrapper functions for check_grad
    def cost_flattened(W_flattened, X, Y):
        W = W_flattened.reshape(-1, 1)  # Reshape W inside
        return cost(X, Y, W).item()

    def cost_der_flattened(W_flattened, X, Y):
        W = W_flattened.reshape(-1, 1)  # Reshape W inside
        return cost_der(X, Y, W).flatten()  # Return flattened gradient

    # Check the gradient
    print('Gradient using scipy'.center(100, '*'))
    error = check_grad(cost_flattened, cost_der_flattened, W.flatten(), X, Y)
    print("Gradient check error:", error)
