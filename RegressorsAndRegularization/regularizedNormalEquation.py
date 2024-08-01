import numpy as np
from sklearn.base import BaseEstimator
from utility import train_test_split_data
from sklearn import metrics


class NormalEquationWithRdige(BaseEstimator):
    def __init__(self, alpha):
        self.alpha = alpha
        super().__init__()

    def fit(self, X: np.array, y: np.array):
        self.model: np.array = np.linalg.inv(X.T @ X + self.alpha * np.identity(X.shape[1])) @ X.T @ y
        return self

    def predict(self, X):
        return X @ self.model.T

    def coef_(self):
        return self.model.tolist()


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = train_test_split_data()

    for alpha in [0, 0.1, 1, 10, 100, 10000]:
        model = NormalEquationWithRdige(alpha=alpha)
        model.fit(x_train, y_train)

        error = metrics.mean_squared_error(model.predict(x_test), y_test)
        print(f'{alpha=}'.center(100, '-'))
        print(f'{error=}')
        print(f'{model.coef_}')
