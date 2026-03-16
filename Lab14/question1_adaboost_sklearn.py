import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def load_data():
    iris = load_iris()
    X = iris.data
    y = iris.target
    y=np.where(y==0,0,1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=999)
    return X_train, X_test, y_train, y_test

def scaling_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s


def model(X_train, X_test, y_train):
    base_model = DecisionTreeClassifier(max_depth=1)
    clf = AdaBoostClassifier(estimator=base_model, n_estimators=10)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred

def evaluating_model(y_pred,y_test):
    acc_score = accuracy_score(y_test,y_pred)
    classfn_report = classification_report(y_test,y_pred)
    return acc_score,classfn_report


def main():
    X_train, X_test, y_train, y_test = load_data()
    print("Train set:\n", X_train)
    print("\nTest set:\n", X_test)
    X_train_scaled, X_test_scaled = scaling_data(X_train, X_test)
    y_pred = model(X_train_scaled, X_test_scaled, y_train)
    acc_score, classfn_report = evaluating_model(y_pred,y_test)
    print("\nAccuracy Score:",acc_score)
    print("\nClassification Report:",classfn_report)


if __name__ == '__main__':
    main()
