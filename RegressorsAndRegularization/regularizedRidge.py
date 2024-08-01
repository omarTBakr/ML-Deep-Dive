import numpy as np

from utility import train_test_split_data
from sklearn import linear_model, preprocessing, model_selection
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


model = linear_model.Ridge()
param_grid = {'alpha': [0.01, 0.1, 1, 10, 100, 1000, 10000], 'fit_intercept': [True, False]}
kf = model_selection.KFold(n_splits=4)
search = model_selection.GridSearchCV(model, param_grid=param_grid, cv=kf, scoring='neg_root_mean_squared_error')

# get the data
x_train, x_test, y_train, y_test = train_test_split_data()
poly = preprocessing.PolynomialFeatures(degree=2)
poly.fit(x_train)
search.fit(poly.transform(x_train), y_train)


best_error = search.best_score_
print(f'{-1*best_error=}')
print(search.best_estimator_)
# results = search.cv_results_['mean_test_score']
# results =np.mean(results.reshape(2,7), axis=0)
# plt.plot( np.log10(param_grid['alpha'] ) ,
#           results)
# plt.show()
