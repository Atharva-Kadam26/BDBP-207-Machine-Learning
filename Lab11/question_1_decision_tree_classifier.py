
import numpy as np
import pandas as pd
from sklearn.decomposition import non_negative_factorization
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

def load_data():
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y

def entropy(y):
    values, counts = np.unique(y, return_counts=True)
    probability = counts / len(y)
    return -1 * (np.sum(probability * np.log2(probability)))


def information_gain(y,left,right):

    parent_entropy = entropy(y)
    w_left = len(left) / len(y)
    w_right = len(right) / len(y)

    child_entropy = w_left * entropy(left) + w_right * entropy(right)
    gain = parent_entropy - child_entropy

    return gain

def build_tree(X,y,k):

    if len(X) <= k:                           # if node has few samples we stop spiltting
        return np.bincount(y).argmax()       #finds majority class

    m , n = X.shape

    best_feature = None
    best_split = None
    best_gain = -float('inf')

    for i in range(n):    #try every feature
        values = np.unique(X[:,i])
        splits = (values[:-1] + values[1:]) / 2

        for split in splits:

            left = X[:,i] <= split
            right = X[:,i] > split

            if sum(left)==0 or sum(right)==0:
                continue

            y_left = y[left]
            y_right = y[right]

            gain = information_gain(y,y_left, y_right)

            if gain > best_gain:
                best_gain = gain
                best_feature = i
                best_split = split


    if best_feature is None:
        return np.bincount(y).argmax()

    left_mask = X[:,best_feature] <= best_split
    right_mask = X[:,best_feature] > best_split

    left_tree = build_tree(X[left_mask],y[left_mask],k)
    right_tree = build_tree(X[right_mask],y[right_mask],k)

    return (best_feature, best_split , left_tree, right_tree)


#Predicting one sample
def predict_sample(x,tree):

     if not isinstance(tree,tuple):
         return tree

     feature , split , left_tree, right_tree = tree

     if x[feature] <= split:
         return predict_sample(x,left_tree)

     else:
         return predict_sample(x,right_tree)


#Predict Dataset
def predict_dataset(X,tree):
    return np.array([predict_sample(x,tree) for x in X])


def main():
    X,y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=999)
    tree = build_tree(X_train, y_train, 60)
    y_pred = predict_dataset(X_test,tree)
    print("Predictions:",y_pred)
    print("Ground Truth y values:",y_test)
    print("Accuracy:",accuracy_score(y_test,y_pred))



if __name__ == '__main__':
    main()
