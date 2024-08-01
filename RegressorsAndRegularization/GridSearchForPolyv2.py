from sklearn.base import BaseEstimator
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from utility import train_test_split_data
from sklearn.metrics import mean_squared_error
class SpecialPoly(BaseEstimator):

    def __init__(self, /,alpha , degree , fit_intercept):
        self.alpha = alpha
        self.degree = degree
        self.poly = PolynomialFeatures(degree=degree)
        self.fit_intercept = fit_intercept

    def fit(self , x, y):
        self.poly = self.poly.fit(x)
        x_transformed = self.poly.transform(x)
        self.model = Ridge(alpha=self.alpha,
                           fit_intercept=self.fit_intercept
                           )
        self.model.fit(x_transformed,y)

        return self

    def predict(self , x):
        return self.model.predict(
            self.poly.transform(x)
        )



if __name__ == "__main__":
    x_train,x_test,y_train,y_test = train_test_split_data()
    estimator = SpecialPoly(alpha=0.1 , degree=1, fit_intercept=False)
    estimator.fit(x_train,y_train)
    error = mean_squared_error(
        estimator.predict(x_test) ,
        y_test

    )

    print(f'{error=}')