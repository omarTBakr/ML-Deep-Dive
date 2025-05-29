from sklearn import neural_network
from sklearn import metrics
from matplotlib import pyplot as plt
from Data.load_data import load_data
from collections import  namedtuple

best_paramters = {'activation': 'relu',
                  'early_stopping': False,
                  'learning_rate': 'adaptive',
                  'solver': 'adam'}


x_train,x_val, y_train, y_val = load_data()
y_train , y_val= y_train.ravel() , y_val.ravel()

iterations =[]


for i in [100, 70, 50, 35, 20, 10,7,4, 3, 2]:
        NN = neural_network.MLPRegressor(**best_paramters ,
                                         hidden_layer_sizes=i)
        NN.fit(x_train,y_train)
        error = metrics.mean_squared_error(
            NN.predict(x_val),y_val
        )
        iterations.append(  error )


plt.plot([100, 70, 50, 35, 20, 10,7,4, 3, 2] ,iterations )
plt.show()