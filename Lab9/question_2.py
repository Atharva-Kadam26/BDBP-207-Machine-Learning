"""Implement a regression decision tree algorithm using scikit-learn for the simulated dataset."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

def load_data(filename):
    data = pd.read_csv(filename)
    X = data[['age','BMI','BP','blood_sugar','Gender']]
    y = data['disease_score']
    return X, y

def splitting_data(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=999)
    return X_train, X_test, y_train, y_test

def model(X_train,y_train):
    model = DecisionTreeRegressor(random_state=42)
    mod=model.fit(X_train,y_train)
    return mod

def prediction(X_test,mod):
    y_pred = mod.predict(X_test)
    return y_pred

def evaluate_model(y_test,y_pred):
    mse=mean_squared_error(y_test,y_pred)
    return mse



def main():
    X,y = load_data("simulated_data_multiple_linear_regression_for_ML.csv")
    X_train, X_test, y_train, y_test = splitting_data(X,y)
    mod = model(X_train,y_train)
    y_pred = prediction(X_test,mod)
    mse = evaluate_model(y_test,y_pred)
    print("Predicted y:",y_pred)
    print("Actual y:",y_test.values)
    print("MSE:",mse)


if __name__ == '__main__':
    main()