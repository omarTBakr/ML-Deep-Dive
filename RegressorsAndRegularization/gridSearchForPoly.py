from sklearn import preprocessing
from sklearn import model_selection, linear_model
from sklearn.pipeline import Pipeline
from utility import train_test_split_data


pipeline = Pipeline(
    [
        ('scaler', preprocessing.MinMaxScaler()),
        ('Poly', preprocessing.PolynomialFeatures(include_bias=False)),
        ('Ridge', linear_model.Ridge()),
    ]
)
p_grid = {'Ridge__alpha': [1, 0.1, 10],
          'Poly__degree': [1, 2, 3],
          'Ridge__fit_intercept': [True, False]}

search = model_selection.GridSearchCV(pipeline, param_grid=p_grid, cv=5, scoring='neg_mean_squared_error')


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = train_test_split_data()

    search.fit(x_train, y_train)
    print(f'{search.best_score_}')
    print(f'{search.best_estimator_}')
