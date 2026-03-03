"""Compute SONAR classification results with and without data pre-processing (data
normalization). Perform data pre-processing with your implementation and with
scikit-learn methods and compare the results"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def load_data(filename):
    df = pd.read_csv(filename)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

def split_data(X,y):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=999)
    return X_train,X_test,y_train,y_test

def normalize_data(X_train,X_test):
    x_max=np.max(X_train,axis=0)
    x_min=np.min(X_train,axis=0)
    X_train_normal=(X_train-x_min)/(x_max-x_min)
    X_test_normal=(X_test-x_min)/(x_max-x_min)

    return X_train_normal,X_test_normal

def model_logistic(X_train_normal,y_train):
    model=LogisticRegression(max_iter=2000)
    model.fit(X_train_normal,y_train)
    return model

def y_pred(model,X_test_normal):
    y_pred = model.predict(X_test_normal)
    return y_pred


def accuracy(y_test,y_pred):
    accur = accuracy_score(y_test,y_pred)
    return accur



def main():
    X,y=load_data('sonar data.csv')
    X_train, X_test, y_train,y_test = split_data(X,y)
    X_train_normal,X_test_normal=normalize_data(X_train,X_test)
    mod=model_logistic(X_train_normal,y_train)
    y_pr=y_pred(mod,X_test_normal)
    accur=accuracy(y_test,y_pr)
    print(accur)




if __name__ == "__main__":
    main()