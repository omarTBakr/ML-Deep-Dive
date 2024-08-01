from data_utility import get_data_split_numpy
from sklearn import model_selection, preprocessing, linear_model
from utility import calc_RMSE
from data_utility import get_data_split_numpy
from utility import train_test_split_data


if __name__ == '__main__':
    model = linear_model.LinearRegression(fit_intercept=True)
    x, y = get_data_split_numpy()

    model.fit(x, y)

    x_train, x_test, y_train, y_test = train_test_split_data()
    base_line_error = calc_RMSE(model, x_test, y_test)

    # print(f'{base_line_error=}') #5.494044038918205

    scaler = preprocessing.MinMaxScaler()
    scaler.fit(x_train)

    model.fit(scaler.transform(x_train), y_train)

    error_after_normalization = calc_RMSE(model, scaler.transform(x_test), y_test)

    # print(f'{error_after_normalization=}') #12.216701916378886
