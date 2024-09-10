import numpy as np
from numpy.linalg import norm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , classification_report


def load_breast_cancer_scaled():
    data = load_breast_cancer()
    X, t = data.data, data.target_names[data.target]
    t = (t == 'malignant').astype(int)

    scaler  = MinMaxScaler()
    X= scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, t, test_size=0.3,
                                                        shuffle=True, stratify=t,
                                                        random_state=0)
    return X_train, X_test, y_train, y_test , scaler


if __name__ == '__main__':

    X_train, X_test, y_train, y_test ,scaler = load_breast_cancer_scaled()
    model = LogisticRegression(solver='lbfgs')
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    # scale the test data
    # X_test = scaler.fit_transform(X_test)
    y_pred_test = model.predict(X_test)
    y_pred_test_prop = model.predict_proba(X_test)[:, 1]

    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test = accuracy_score(y_test, y_pred_test)

    print('Training accuracy: %.4f' % accuracy_train)
    print('Test accuracy:     %.4f' % accuracy_test)

    report_train = classification_report(y_train, y_pred_train)
    report_test = classification_report(y_test, y_pred_test)
    print('Training\n%s' % report_train)
    print('Testing\n%s' % report_test)