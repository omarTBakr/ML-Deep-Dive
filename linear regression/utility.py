import matplotlib.pyplot as plt
from gradientDecent import gradient_descent_linear_regression
import collections
from tqdm import tqdm
from DataProcessing.load_data import load_data, load_best_feature


def visualize(x, y, *, label='x y ', xlabel='x', ylabel='y', scatter=False):  # plot scatter plot
    plt.figure(label)

    if scatter:
        plt.scatter(x, y, label=label)

    else:
        plt.plot(x, y, label=label)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


# method for testing linear regression over different options
'''

# named tuple to hold all vital hyperparameters for the test 
    
    error 
    optimal weights
    total iterations 
    step size 
    precision 
    procesor used over the data 
'''

Iteration = collections.namedtuple('Iteration', 'error optimal_weights iterations step_size precision processor')


def test_linear_regression(
    X,
    Y,
    W,  # np array of weights to start with
    *,
    processors,  # tuple of different processors to prepocess the data
    step_sizes,  # tuple of different step sizes
    precisions,  # tuple of different precisions
    max_iterations=10_000,
    best_feature=False,
):
    hyperparameters_info = []  # list of the named tupples

    for step_size in tqdm(step_sizes):
        for precision in precisions:
            for processor in processors:
                if processor:
                    X, Y = processor(X, Y)

                cost, optimal_weights, iterations = gradient_descent_linear_regression(
                    X, Y, W, step_size=step_size, precision=precision, max_iter=max_iterations
                )

                hyperparameters_info.append(
                    Iteration(cost[-1], optimal_weights, iterations, step_size, precision, processor)
                )

    return hyperparameters_info
