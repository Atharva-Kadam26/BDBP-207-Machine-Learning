"""Compute SONAR classification results with and without data pre-processing (data
normalization). Perform data pre-processing with your implementation and with
scikit-learn methods and compare the results"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


"""Without pre-processing (normalization)"""


def load_data(filename):
    df = pd.read_csv(filename)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

def split_data(X,y):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=999)
    return X_train,X_test,y_train,y_test

def model_logistic(X_train,y_train):
    model=LogisticRegression()
    model.fit(X_train,y_train)
    return model

def y_pred(model,X_test):
    y_pred = model.predict(X_test)
    return y_pred


def accuracy(y_test,y_pred):
    accur = accuracy_score(y_test,y_pred)
    return accur


def main():
    X, y = load_data('sonar data.csv')
    X_train,X_test,y_train,y_test = split_data(X,y)
    model = model_logistic(X_train,y_train)
    pred_y = y_pred(model,X_test)
    acc=  accuracy(y_test,pred_y)
    print('Accuracy is: ',acc)


if __name__ == '__main__':
    main()




