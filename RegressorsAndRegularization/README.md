# Regressors & Regularization Deep Dive 
## Required packages 
```bash
pip install numpy sklearn pandas 
```
 
## Learning Objectives
 1. hands on concepts on topics such as 
    1. polynomial reagression , I have already a deep dive over it,[here](https://github.com/omarTBakr/ML-Deep-Dive/tree/main/Polynomial%20regression)
    2. cross validation 
 
2. use `sklearn.linear_modle`
3. `Ridge` and `Lasso` regularizations
4. `GridSearchCV` tour 
5.discover the impact of normalizing , standarizing data over the performance 


## Tasks 
1. KISS ( keep it simple and stupid) get a baseline model without any modifications
  - _for simplicity, you don't have to split the data into training , validation and testing at the end of the data 
   it's just a toy dataset_ just use `train_test_split` from `sklearn` and keep `test_size = 0.5`
  - use `root_mean_squared_error` from `sklearn` for evaluation 


2. transform the data using `sklearn.Polynomial` and save `RMSE` for both training and validation , test with degrees `[1,2,3,4]` 
   what are your opeservations
3. implement a function called  `monomials` which only transform each feature to the given power only, it does not take into consideration
  the combination between features like the built-in one in `sklearn` 
   aka you have x ,and degree 3 for example , then you will trnasform it to x , x^2 , x^3 
 
    __try to make it as vectorized as possible__
4. make a function that takes 0 based indexing for the feature and polynomial degree order and plot the error for train vs val in a bar plot 
5. test the performance with Regularization , try `alpha=[0.01, 0.1, 1, 10, 100, 1000, 10000]`, use grid search with k-fold k =4 
6. use lasso for feature selection to get the best 10 features you have ,check `alpah =[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 10]` 
7. implement Regularizing Normal Regression using numpy -make a class called `NormalEQuationsWithRidge` that implements fit and predict and make it
complinat with sklearn if you want
8. update the code for linear regression in this [repo](https://github.com/omarTBakr/ML-Deep-Dive/blob/main/linear%20regression/cost.py) to include regularization

9. Create a program that administrate creating a pipeline / GridSearchCV that
allows learning ridge and grid search on:
        
   1. Polynomials features degree `[1 , 2 ,3]`
   2. Ridge parameters like `alpha` , `fit_intercept`

10. To administrate learning how to extend libraries, let’s create a class that
performs 2 steps together 
## my apporach
_note: if you want to follow up with my solultion , you should fix the random state_
```python

np.random.seed(17)
```
you will find the toy data set in the `Data` also  , the data is cleaned no feature engineering is required
1. make a Data directory to hold the data with in
2. make a simple `data_utility.py` to handle all data requirements
3. make a function to load the data 
```python 
 def get_data_split_numpy():
    data = get_numpy()

    return data[:, :-1], data[:, -1]

```
4. get the data and split it 
```python 

x , y = get_data_split_numpy()
x_train,x_test , y_train,y_test = model_selection.train_test_split(
                            x,y ,
                            shuffle= False ,
                            random_state=17 , test_size=0.5)
 
```
5. for the `LinearRegression` without normalization 
```python
model = linear_model.LinearRegression(fit_intercept=True)
model.fit(x,y)
base_line_error = metrics.root_mean_squared_error(
    model.predict(x_test) ,
    y_test
                                )
```
6. for normalization
here are the steps
   1. make the scaler fit the data 
   2. use the scaler to transform the the data 
   __train and evaluation data should be done separately to avoid any data leakage__


```python
scaler = preprocessing.MinMaxScaler()
scaler.fit(x_train)


model.fit(scaler.transform(x_train), y_train)

error_after_normalization = metrics.root_mean_squared_error(
    model.predict(scaler.transform(x_test) ) ,
    y_test
)
```
7. make a file contains all `Polynomial` trials just for organization purposes  
8. make a simple utility function called `fit_polynomial` 
   here is a smaple 
```python 
def fit_polynomial(model, x, y, degree=1):
    
    poly = preprocessing.PolynomialFeatures(degree=degree)
    poly.fit(x)
    model.fit(poly.transform(x), y)
    return model, poly
 
```
9. plot the train , validation error vs the degree order 


 ![image](irregularizedPolynomial.png)

__notice how the model is overfitting the data after degree and the performance degregation due to that__

10. make another module for `monomial.py` just for the sake of organization
11. here are my implementations for the function `monomial_features`
```python
def monomail_features(degree,x:np.array):
    transformed = x
    for degree in range(2,degree+1):
        transformed = np.hstack((transformed,np.power(x,degree)))

    return transformed


```
__and here is the error curve__

![image](monomial.png)


**you can observe that the model is overfitting the data just after the second monomial degree**

12. the function is exactly the same except for the plotting part , here is my approach to it 
```python 
def test_individual_features(*features, degree=None):
    if degree is None:
        degree = [1, 2, 3]
    x_train, x_test, y_train, y_test = train_test_split_data()
    model =linear_model.LinearRegression(fit_intercept=True)
    train_errors,test_errors = [] ,[]

    for feature  in features:
        for degree in degrees:
            selected_train  , selelcted_test= x_train[:,feature].reshape(-1,1), x_test[:,feature].reshape(-1,1)

            model, poly = fit_polynomial(model, selected_train, y_train, degree)
            train_errors.append(calc_RMSE(
                model, poly.transform(selected_train), y_train
            ))
            test_errors.append(
                calc_RMSE(model, poly.transform(selelcted_test), y_test)
            )


    return train_errors, test_errors

```
__and the plotting__
```python
    degree =[1]
    features = [ 0 ,3 ,6 ]
    train, test  = test_individual_features(*features , degree= degrees)

    print(f'{train=}\n,{test=}')
    plt.figure()
    bar_width = 0.25
    plt.bar(features , train ,width= bar_width , color ='r',
            label ='training')
    plt.bar([feature + bar_width for feature in features], test, width=bar_width, color='b',
            label='testing')

    plt.xlabel('Features',fontweight ='bold', fontsize = 15)

    plt.ylabel('Error', fontweight='bold', fontsize=15)

    plt.legend(loc = 'best')
    plt.show()

```
![image](individualFeatures.png)
13. again make a module for regularization  
 here is my apporach 
 1. make the grid search which contains all the hyperoparameters you want to search over 
2. make the kfold or you can directly use `cv =number_of_split`
3. put it all  together in `GridSearchCv`
4. fit the esitimator , and you can infer it to get the best performance indicators 

```python

from utility import train_test_split_data
from sklearn import linear_model,preprocessing,model_selection
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


model = linear_model.Ridge()
param_grid = {'alpha':[0.01, 0.1, 1, 10, 100, 1000, 10000],
              'fit_intercept':[True , False]
              }
kf = model_selection.KFold(n_splits=4)
search =model_selection.GridSearchCV(
    model , param_grid= param_grid ,
    cv = kf , scoring ='neg_root_mean_squared_error'
)

# get the data
x_train,x_test , y_train,y_test = train_test_split_data()
poly =  preprocessing.PolynomialFeatures(degree=2)
poly.fit(x_train)
search.fit(poly.transform(x_train),y_train)

```
14. this one was a bit tricky , using lasso for feature selection , you can divide this into two parts
 1. is to find the best model 
2. select the best n features according to that model
 __the first part is almost identical to the previous task__

```python

x_train,x_test, y_train, y_test = train_test_split_data()

param_grid  = {'alpha':[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 10] ,
               }
model = linear_model.Lasso(fit_intercept=True)


search = model_selection.GridSearchCV( model , param_grid=param_grid ,
                                       cv = 4 ,
                                       scoring='neg_root_mean_squared_error')

search.fit(x_train,y_train)
print(f'{search.best_estimator_=}, {-1*search.best_score_=}')
```
__the second part is to choose the best 10 features__
_here is my approach and it could be implemented in a more nicer way , but here we are!_

```python
selector = feature_selection.SelectFromModel(
    estimator= search.best_estimator_ ,

).fit(x_train,y_train)
# print(dir(selector))
boolean_support = selector.get_support()
print([index for index,value in  enumerate(boolean_support) if value][:10])

```
17. implement normal equation regularized should be straightforward
here is the equation 

![img.png](img.png)
```python 
class NormalEquationWithRdige(BaseEstimator):
    def __init__(self, alpha):
        self.alpha = alpha
        super().__init__()

    def fit(self, X:np.array,y:np.array):
        self.model :np.array =np.linalg.inv(X.T@X+self.alpha* np.identity(X.shape[1])
                                  )@X.T@y
        return self

    def predict(self,X):
        return X@self.model.T

    def coef_(self):
        return self.model.tolist()


```
18. updating the un regularized linear regression
 this should be trivial since you will only add alph/2 W^2 to the cost function and update the derivative w.r.t. weights

```python
import numpy as np


def cost(X: np.array, Y: np.array, W: np.array, alpha: float):
    # add new column of ones to X
    # X_new =np.hstack((np.ones(( X.shape[0] , 1)) , X) )
    N = X.shape[0]
    # return np.sum(
    #     np.square(error(X ,Y,W))
    #         )/N/2
    _error = error(X, Y, W)
    return (_error.T @ _error) / N / 2 + alpha / 2 * np.square(W)

def cost_der(X: np.array, Y: np.array, W: np.array,alpaha: float):
    X_new = np.hstack((np.ones((X.shape[0], 1)), X))
    N = X.shape[0]
    return X_new.T @ error(X, Y, W) / N + alpha*W

```
19 . there is a suddle point with making a pipeline with those requirements which use you should appreviate 
the parameters for the grid search with the name of the corresponding variable likke below 
```python 
pipeline = Pipeline(
    [
        ('scaler', preprocessing.MinMaxScaler()),
        ('Poly', preprocessing.PolynomialFeatures(include_bias=False)),
        ('Ridge', linear_model.Ridge()),
    ]
)
p_grid = {'Ridge__alpha': [1, 0.1, 10],
          'Poly__degree': [1, 2, 3],
          'Ridge__fit_intercept': [True, False]}

```

20. extend again from `BaseEstimator` and implement `fit` and `predict` 
```python

class SpecialPoly(BaseEstimator):

    def __init__(self, /,alpha , degree , fit_intercept):
        self.alpha = alpha
        self.degree = degree
        self.poly = PolynomialFeatures(degree=degree)
        self.fit_intercept = fit_intercept

    def fit(self , x, y):
        self.poly = self.poly.fit(x)
        x_transformed = self.poly.transform(x)
        self.model = Ridge(alpha=self.alpha,
                           fit_intercept=self.fit_intercept
                           )
        self.model.fit(x_transformed,y)

        return self

    def predict(self , x):
        return self.model.predict(
            self.poly.transform(x)
        )



```
