
# Binary Classification with Log Loss

In this task:

- We will take a deep dive into binary classification and see the inner details of the algorithm by implementing it from scratch.
- Have a look inside focal loss and implement it as well.

## Tasks

1. **Implement a binary classifier** [simple changes to this code here](https://github.com/omarTBakr/ML-Deep-Dive/tree/main/linear%20regression)
2. **Test that binary classifier** on the breast cancer dataset from sklearn. 
   _Here is a script to load this dataset and normalize it:_

   ```python
   from sklearn.datasets import load_breast_cancer
   from sklearn.preprocessing import MinMaxScaler
   from sklearn.model_selection import train_test_split

   def load_breast_cancer_scaled():
       data = load_breast_cancer()
       X, t = data.data, data.target_names[data.target]
       t = (t == 'malignant').astype(int) 

       scaler  = MinMaxScaler()
       X = scaler.fit_transform(X)

       X_train, X_test, y_train, y_test = train_test_split(X, t, test_size=0.3,
                                                           shuffle=True, stratify=t,
                                                           random_state=0)
       return X_train, X_test, y_train, y_test, scaler
   ```

3. **Implement focal loss** [here is an article about it](https://towardsdatascience.com/focal-loss-a-better-alternative-for-cross-entropy-1d073d92d075)

## My Approach

1. Make a `sklearn_lib.py` file to test the performance of the `sklearn.linear_model.LogisticRegression` model.
2. **Modify the `cost` function:**

   ```python
   def cost_f(X, t, weights):
       pass  # Implement log loss or focal loss here
   ```

3. **Make a `sigmoid` function:**

   ```python
   def sigmoid(x):
       pass  # Implement the sigmoid function here
   ```

   __Note:__ The derivative of log loss is the same as Mean Squared Error, thus you don't have to change the derivative function from the old code. 
   **Hint:** _Make sure this note about the derivative is accurate. It might need some adjustments._

4. **Implement the focal loss function:**

   ```python
   import numpy as np  

   def focal_loss_with_class_weight(y_true, y_pred, alpha=0.25, gamma=2.0):
       """
       Compute the focal loss for a binary classification problem.
       """
       return alpha*-(1-y_pred)**gamma*y_true*np.log(y_pred) - (1-alpha)*(1-y_true)*y_pred**gamma*np.log(1-y_pred)
   ```

   Here is a breakdown:
     1. `y_true * np.log(y_pred)`: This is the cost for positive examples.
     2. `(1 - y_true) * np.log(1 - y_pred)`: Loss for negative examples.
     3. `alpha` is just a factor controlling the balance between positive and negative examples. 
     4. `(1 - y_pred)**gamma`: This is the modulating factor of focal loss; when `y_pred` goes to 1, this term will go to zero, reducing the penalty for well-classified positive examples.
     5. `y_pred**gamma`: This term goes to zero as `y_pred` goes to 0, reducing the penalty for well-classified negative examples.

 
