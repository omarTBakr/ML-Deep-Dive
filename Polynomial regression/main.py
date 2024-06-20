import numpy as np

from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import collections
from tqdm import tqdm
from data import generate_uniform_data,get_some_insights
from utility import  visualize2d , pair_visualization ,visualize_trails
import seaborn as sns



Trial = collections.namedtuple('Trial', 'model_degree  error  N model poly')



def fit_polynomyal( X,Y,degree):

    '''
        input :
          X ,Y data  we want to train th model over
          degree: the maximum degree of the polynomyial and train a model accordingly
        return :
            mean squared error
            optimal coefficients
            number of training examples

    '''
    liner_reg = LinearRegression()
    poly = PolynomialFeatures(degree=degree, include_bias=True)

    X_new = poly.fit_transform(X) # transfer the data to include polynomial features
    liner_reg.fit(X_new, Y)
    error = mean_squared_error(liner_reg.predict(X_new), Y) / 2
    return error, X.shape[0] , liner_reg , poly


def try_polynomials(X,Y , start=1, end=4):
    '''

    :param X: dependent variables
    :param Y:   independent variables
    :param start:  start range of polynomials
    :param end:  end range of polynomials
    :return:  list of named tuples each one contains del_degree  error  coef  n(traning examples)
    '''

    return [  Trial(degree, *fit_polynomyal(X,Y,degree))
              for degree in tqdm( range(start, end+1))
              ]


if __name__ == '__main__':
    func = lambda x: 5 + x + 10 * x ** 2 + 5 * x ** 3
    X,Y = generate_uniform_data(func)
    # get_some_insights()

    # let's visualize the data
    # visualize2d(X,Y)
    # visualize2d(X,func(X),scatter=False)
    pair_visualization(X,Y , X,func(X))

    start, end = 1, 14
    trials_info = try_polynomials(X,Y,start, end)

    # visualize degree vs error

    visualize2d(x=list(range(start, end + 1)),
                y=[trial.error for trial in trials_info],
                scatter=False,
                title='degree vs error for $5 + x + 10 x^2 + 5x^3$')



    visualize_trails(X,Y, trials_info)
    #(BIC) baysian  information characteristic
    visualize2d(x=list(range(start, end + 1)),
                y=[
                    trial.N * np.log(trial.error) + k * np.log(trial.N)
                    for trial, k in zip(trials_info, range(start, end + 1))
                ],
                scatter=False,
                title=r'degree vs BIC =$n \times \log{SS} +k\log{n}$'
                )

    ###
