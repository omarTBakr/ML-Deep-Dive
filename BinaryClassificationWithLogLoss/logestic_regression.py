import numpy as np
# from mpmath import sigmoid
from numpy.linalg import norm
from sklearn.metrics import classification_report , accuracy_score
from sklearn_lib import load_breast_cancer_scaled

def sigmoid(x):
    return 1/(1+np.exp(-x))

def cost_f(X, t, weights):
    examples = X.shape[0]
    x = np.dot(X, weights)

    cost =-t* np.log(sigmoid(x)) - (1-t)*np.log(1- sigmoid(x))
    return cost/examples


def f_derivative(X, t, weights):
    examples = X.shape[0]
    pred = sigmoid(np.dot(X, weights))
    error = pred - t
    gradient = X.T @ error / examples
    return gradient


def gradient_descent_linear_regression(X, t, step_size=0.01, precision=0.0001, max_iter=100000):
    examples, features = X.shape
    iter = 0
    cur_weights = np.random.rand(features)  # random starting point
    last_weights = cur_weights + 100 * precision  # something different

    # print(f'Initial Random Cost: {cost_f(X, t, cur_weights)}')

    while norm(cur_weights - last_weights) > precision and iter < max_iter:
        last_weights = cur_weights.copy()  # must copy
        gradient = f_derivative(X, t, cur_weights)
        cur_weights -= gradient * step_size
        # print(cost_f(X, cur_weights))
        iter += 1

    # print(f'Total Iterations {iter}')
    # print(f'Optimal Cost: {cost_f(X, t, cur_weights)}')
    return cur_weights


if __name__ == '__main__':
    # np.random.seed(0)  # If you want to fix the results

    X_train, X_test, y_train, y_test, scaler = load_breast_cancer_scaled()
    optimal_weights = gradient_descent_linear_regression(X_train, y_train)

    y_pred_train = sigmoid(np.dot(X_train , optimal_weights)).reshape(-1 , 1)>=0.5
    y_pred_test = sigmoid(np.dot(X_test, optimal_weights)).reshape(-1 , 1)>=0.5


    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test = accuracy_score(y_test, y_pred_test)

    print('Training accuracy: %.4f' % accuracy_train)
    print('Test accuracy:     %.4f' % accuracy_test)

    report_train = classification_report(y_train, y_pred_train)
    report_test = classification_report(y_test, y_pred_test)
    print('Training\n%s' % report_train)
    print('Testing\n%s' % report_test)

