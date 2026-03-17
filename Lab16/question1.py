"""Write a Python program to aggregate predictions from multiple trees to output a final
prediction for a regression problem."""
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
import numpy as np

def data():
    dat = load_diabetes()
    X = dat.data
    y = dat.target
    return X,y

def splitting_data(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 999)
    return X_train, X_test, y_train, y_test

def creating_trees(X_train,X_test,y_train,num_of_trees):
    models=[]
    learning_rate = 0.1

    mean_y = sum(y_train)/len(y_train)   #Initial prediction(mean)
    y_pred = [mean_y] * len(y_train)

    for i in range(num_of_trees):
        residuals = y_train - y_pred
        for j in range(len(y_train)):
            
    for x in X_test:
        total =0
        for tree in models:
            total += 0.1 * tree.predict([x])[0]   #0.1 = lambda

        y_pred.append(total)


