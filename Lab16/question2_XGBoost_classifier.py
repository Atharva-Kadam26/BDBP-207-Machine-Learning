"""Implement XGBoost classifier using scikit-learn"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

def load_data():
    iris = load_iris()
    X = iris['data']
    y = iris['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 999)
    return X_train, X_test, y_train, y_test

def scaling_data(X_train,X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def model(X_train,y_train):
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    trained_model = model.fit(X_train, y_train)
    return trained_model

def predict(trained_model,X_test):
    y_pred = trained_model.predict(X_test)
    return y_pred

def evaluating_model(y_test,y_pred):
    accuracy = accuracy_score(y_test,y_pred)
    return accuracy

def main():
    X_train, X_test, y_train, y_test = load_data()
    X_train_s,X_test_s = scaling_data(X_train,X_test)
    trained_model = model(X_train_s,y_train)
    y_pred = predict(trained_model,X_test_s)
    accuracy = evaluating_model(y_test,y_pred)
    print("Accuracy:",accuracy)



if __name__ == '__main__':
    main()

