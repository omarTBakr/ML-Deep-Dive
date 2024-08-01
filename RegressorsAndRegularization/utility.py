from sklearn import metrics, model_selection
from data_utility import get_data_split_numpy


def calc_RMSE(model, x, y):
    return metrics.root_mean_squared_error(model.predict(x), y)


def train_test_split_data():
    x, y = get_data_split_numpy()
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        x, y, shuffle=False, random_state=17, test_size=0.5
    )

    return x_train, x_test, y_train, y_test
