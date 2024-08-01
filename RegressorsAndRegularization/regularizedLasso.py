from sklearn import linear_model, preprocessing, model_selection, feature_selection
from utility import train_test_split_data
import warnings

warnings.filterwarnings("ignore")


x_train, x_test, y_train, y_test = train_test_split_data()

param_grid = {
    'alpha': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 10],
}
model = linear_model.Lasso(fit_intercept=True)


search = model_selection.GridSearchCV(model, param_grid=param_grid, cv=4, scoring='neg_root_mean_squared_error')

search.fit(x_train, y_train)
print(f'{search.best_estimator_=}, {-1*search.best_score_=}')
selector = feature_selection.SelectFromModel(
    estimator=search.best_estimator_,
).fit(x_train, y_train)
# print(dir(selector))
boolean_support = selector.get_support()
print([index for index, value in enumerate(boolean_support) if value][:10])
