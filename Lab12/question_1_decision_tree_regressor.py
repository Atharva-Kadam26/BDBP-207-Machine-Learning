"""Implement decision tree classifier without using scikit-learn using the iris dataset. Fetch the iris dataset from scikit-learn library."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score

def load_data():
    iris = load_diabetes()
    X = iris.data
    y = iris.target
    return X, y

#Build Tree
def build_tree(X, y , k):

    if len(X) <= k:
        return float(np.mean(y))

    m,n = X.shape   # m = no. of samples , n = no. of features
    best_feature = None
    best_split = None
    best_error = float('inf')     #inf meaning the initial error is large

    for i in range(n):            #checking every feature
        values = np.unique(X[:,i])

        splits = (values[:-1] + values[1:]) / 2

        #testing all the split values
        for split in splits:
            left = X[:,i] <= split
            right = X[:,i] > split

            if sum(left) ==0 or sum(right) ==0:
                continue

            y_left = y[left]
            y_right = y[right]

            y_hat_left = np.mean(y_left)
            y_hat_right = np.mean(y_right)

            error_left = float(np.sum((y_left - y_hat_left)**2))
            error_right = float(np.sum((y_right - y_hat_right)**2))

            total_error = error_left + error_right

            if total_error < best_error:    # argmin(E,s)
                best_error = total_error
                best_feature = i
                best_split = float(split)

    if best_feature is None:
        return float(np.mean(y))

    left_child = X[:,best_feature] <= best_split
    right_child = X[:,best_feature] > best_split

    left_tree = build_tree(X[left_child], y[left_child], k)
    right_tree = build_tree(X[right_child], y[right_child], k)

    return (best_feature, best_split,left_tree, right_tree)

def predict_sample(x,tree):

    if not isinstance(tree, tuple):    #isinstance(x,int) checks whether the variable matches the data type we have given
        return tree                    # a node is just a single value and not a tuple so if it encounters a node return prediction directly

    feature,split,left_tree,right_tree = tree

    if x[feature] <= split:
        return predict_sample(x,left_tree)

    else:
        return predict_sample(x,right_tree)

def predict_dataset(X,tree):
    return np.array([predict_sample(x,tree) for x in X])



def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    tree = build_tree(X_train, y_train, 60)
    y_pred = predict_dataset(X_test,tree)
    mse = np.mean((y_pred-y_test)**2)

    print("Predictions:", y_pred)
    print("Ground truth y values:", y_test)
    print("MSE:", mse)
    print("R2 score:",r2_score(y_test, y_pred))

if __name__ == '__main__':
    main()

