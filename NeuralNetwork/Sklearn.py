import numpy as np
from sklearn import neural_network
from sklearn import model_selection
from Data.load_data import load_data
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")

parameters = {
    'activation': np.array(['relu', 'tanh', 'logistic']),
    'solver': np.array(['adam', 'sgd', 'lbfgs']),
    'learning_rate': np.array(['constant', 'adaptive']),
    'hidden_layer_sizes': np.array([100, 70, 50, 35, 20, 10, 3, 2]),
    'early_stopping': np.array([False, True]),
}


NN = neural_network.MLPRegressor(max_iter=10_000)


kf = model_selection.KFold(n_splits=3)


search = model_selection.GridSearchCV(NN, cv=kf, param_grid=parameters, scoring='neg_mean_squared_error')


x_train, x_val, y_train, y_val = load_data()

search.fit(x_train, y_train.ravel())

if __name__ == '__main__':
    # retaing the model over the whole data
    model = search.best_estimator_
    print('model from cv score over val'.center(50, '#'))
    print(metrics.mean_squared_error(model.predict(x_val), y_val))
    model.fit(x_train, y_train.ravel())

    print('after training over the whole traning set'.center(50, '#'))
    print(metrics.mean_squared_error(model.predict(x_val), y_val))
    print(f"best score from cv is {-1*search.best_score_=}".center(50, '#'))

    print('best paramters are '.center(50, '#'))
    print(search.best_params_)
