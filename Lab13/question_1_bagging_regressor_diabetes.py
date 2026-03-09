"""Implement bagging regressor  using scikit-learn. Use diabetes and iris datasets."""

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import r2_score



def load_data():
    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=999)
    return X_train, X_test, y_train, y_test

def train_bagging_regressor(X_train, y_train):
    basic_model = DecisionTreeRegressor()
    model = BaggingRegressor( estimator=basic_model,n_estimators=100, max_samples = 0.8, bootstrap = True, random_state =999)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    return r2

def main():
    X_train, X_test, y_train, y_test = load_data()
    model = train_bagging_regressor(X_train, y_train)
    r2=evaluate_model(model, X_test, y_test)
    print("R2 Score:", r2)

if __name__ == "__main__":
    main()

