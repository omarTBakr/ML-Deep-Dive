from sklearn.preprocessing import minmax_scale, StandardScaler


def minmax_scaler(X, Y):
    x_shape = X.shape
    X, Y = minmax_scale(X).reshape(x_shape), Y

    return X, Y


def standard_scaler(X, Y):
    x_shape = X.shape
    scaler = StandardScaler()
    X, Y = scaler.fit_transform(X).reshape(x_shape), Y

    return X, Y
