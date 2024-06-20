import matplotlib.pyplot as plt
import numpy as np
from gradientDecent import gradient_descent_linear_regression
from DataProcessing.preprocessing import minmax_scaler, standard_scaler
from DataProcessing.load_data import load_best_feature, load_data

from utility import visualize, test_linear_regression


if __name__ == '__main__':
    W = np.array([1, 1, 1, 1]).reshape(-1, 1)

    # optimizing hyperparameters
    step_sizes = (0.1, 0.01, 0.001, 0.0001, 0.00001, 0.0000001)
    precisions = (0.01, 0.001, 0.0001, 0.00001)
    processors = (minmax_scaler, standard_scaler)
    X, Y = load_data()
    hyperparameters_info_all_features = test_linear_regression(
        X, Y, W, processors=processors, step_sizes=step_sizes, precisions=precisions, max_iterations=10_000
    )

    # investegating hyperparameters
    print(' using all features'.center(100, '%'))

    errors = [iteration.error for iteration in hyperparameters_info_all_features]
    print("minimum error is ".center(100, '$'))

    # interested in minimum error
    minimum_error = np.min(errors)
    print(minimum_error)

    print('best hyperparameters will be '.center(100, '$'))
    index = errors.index(minimum_error)
    print(hyperparameters_info_all_features[index])

    # we will train the model using the best hyperparameters we got
    best_hyperparameters = hyperparameters_info_all_features[index]
    cost, optimal_weights, iterations = gradient_descent_linear_regression(
        *best_hyperparameters.processor(*load_data()),
        step_size=best_hyperparameters.step_size,
        precision=0.1 * best_hyperparameters.precision,
        max_iter=10_000,
    )

    # visualizing the error for our best model

    visualize(
        np.arange(len(cost)),
        cost,
        label=f'Error vs iterations for {iterations} iterations with pre processing '
        f'{best_hyperparameters.processor.__name__} for all features',
        xlabel=' iteration',
        ylabel='Error',
    )

    ########## training the model over the best feature only################

    print(" using the best feature".center(100, '%'))
    processors_bst_feature = (minmax_scaler, standard_scaler)
    W_bst_feature = np.array([1, 1]).reshape(-1, 1)
    X, Y = load_best_feature()
    hyperparameters_info_bst_feature = test_linear_regression(
        X,
        Y,
        W_bst_feature,
        processors=processors_bst_feature,
        step_sizes=step_sizes,
        precisions=precisions,
        max_iterations=10_000,
        best_feature=True,
    )
    # investegating hyperparameters
    print(' using all features'.center(100, '%'))

    errors_bst_feature = [iteration.error for iteration in hyperparameters_info_bst_feature]
    print("minimum error is ".center(100, '$'))

    # interested in minimum error
    minimum_error_bst_feature = np.min(errors_bst_feature)
    print(minimum_error_bst_feature)

    print('best hyperparameters will be '.center(100, '$'))
    index = errors_bst_feature.index(minimum_error_bst_feature)
    print(hyperparameters_info_bst_feature[index])

    best_hyperparameters_bst_feature = hyperparameters_info_all_features[index]

    cost_bst_feature, optimal_weights_bst_feature, iterations_bst_feature = gradient_descent_linear_regression(
        *best_hyperparameters_bst_feature.processor(*load_best_feature()),
        step_size=best_hyperparameters_bst_feature.step_size,
        precision=0.1 * best_hyperparameters_bst_feature.precision,
        max_iter=10_000,
    )

    visualize(
        np.arange(len(cost_bst_feature)),
        cost_bst_feature,
        label=f'Error vs iterations for {iterations} iterations with pre processing for best feature '
        f'{best_hyperparameters_bst_feature.processor.__name__}',
        xlabel=' iteration',
        ylabel='Error',
    )

    # visualize 3 features error features and best feature

    plt.figure('all features Vs best feature ')

    plt.plot(list(range(len(cost_bst_feature))), cost_bst_feature, 'b', list(range(len(cost))), cost, 'r')

    plt.xlabel('iterations')
    plt.ylabel('error')

    plt.legend(['best feature', 'all features'])
    plt.grid()

    plt.show()
