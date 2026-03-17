"""Implement Gradient Boost Regression and Classification using scikit-learn. Use the
Boston housing dataset from the ISLP package for the regression problem and weekly
dataset from the ISLP package and use Direction as the target variable for the
classification."""
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from ISLP import load_data

def loading_data(filename):
    data = load_data(filename)
    X=data.iloc[:,:-1]
    y=data["Direction"]

    #Converting categorical values to numeric
    y.map({"Up":1,"Down":0})
    return X,y

def splitting_data(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=999)
    return X_train, X_test, y_train, y_test

def scaling_data(X_train,X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train,X_test

def model(X_train, y_train):
    gbc_model = GradientBoostingClassifier(n_estimators=100,learning_rate=0.1,max_depth=4,random_state=999)
    trained_model = gbc_model.fit(X_train,y_train)
    return trained_model

def predicting_values(trained_model,X_test):
    y_pred = trained_model.predict(X_test)
    return y_pred

def evaluating_model(y_test,y_pred):
    accuracy = accuracy_score(y_test,y_pred)
    return accuracy

def main():
    X,y = loading_data("Weekly")
    X_train, X_test, y_train, y_test = splitting_data(X,y)
    trained_model = model(X_train, y_train)
    y_pred = predicting_values(trained_model,X_test)
    accuracy = evaluating_model(y_test,y_pred)
    print("Accuracy score:",accuracy)

if __name__ == "__main__":
    main()
