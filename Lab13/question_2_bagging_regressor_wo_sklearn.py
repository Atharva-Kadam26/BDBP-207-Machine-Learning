"""Implement bagging regressor without using scikit-learn"""

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor

def load_data():
    diabetes = load_diabetes()
    X=diabetes.data
    y=diabetes.target
    return X, y

def bootsrapping_samples(X,y):

    num_samples = X.shape[0]    #number of rows in dataset

    indices = np.random.choice(num_samples, num_samples, replace=True)   #random sampling with replacement thats why num_samples

    X_samples = X[indices]    #creating dataset with random indices
    y_samples = y[indices]

    return X_samples, y_samples

def train_bagging_regressor(X_train, y_train,n_estimators=100):    #n_estimators will define the number of models
    models = []
    for i in range(n_estimators):

        X_samples , y_samples = bootsrapping_samples(X_train, y_train)
        model = DecisionTreeRegressor()
        model.fit(X_samples, y_samples)
        models.append(model)

    return models

def predicting(models, X_test):
    predictions = []
    for model in models:
        pred = model.predict(X_test)
        predictions.append(pred)

    predictions = np.array(predictions)
    final_prediction = np.mean(predictions, axis=0)   #we aggregate the output from multiple models
    return final_prediction

def evaluate_model(models, X_test, y_test):
    y_pred = predicting(models, X_test)
    r2 = r2_score(y_test, y_pred)
    return r2

def main():
    X,y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=999)
    models = train_bagging_regressor(X_train, y_train)
    y_pred = predicting(models, X_test)
    r2=evaluate_model(models,X_test,y_test)
    print("Predicted y values are :", y_pred)
    print("Ground truth y values are :", y_test)
    print("Accuracy score is :", r2)

if __name__ == '__main__':
    main()



