"""Implement Random Forest algorithm for regression using scikit-learn. Use diabetes dataset."""

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

def load_data():
    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=999)
    return X_train, X_test, y_train, y_test

def train_random_forest(X_train,y_train):
    model = RandomForestRegressor(n_estimators=100 ,max_depth = None, random_state=999)
    model.fit(X_train, y_train)
    return model

def predict(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred

def evaluate_model(model, X_test, y_test):
    y_pred = predict(model, X_test)
    r2 = r2_score(y_test, y_pred)
    return y_pred, r2

def main():
    X_train, X_test, y_train, y_test = load_data()
    model = train_random_forest(X_train, y_train)
    y_pred = predict(model, X_test)
    r2 = r2_score(y_test, y_pred)
    print("Ground truth y values are :", y_test)
    print("\nPredicted y values are :", y_pred)
    print("\nR2 score:", r2)
if __name__ == '__main__':
    main()
