"""Write a Python program to aggregate predictions from multiple trees to output a final
prediction for a regression problem."""
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


def data():
    dat = load_diabetes()
    X = dat.data
    y = dat.target
    return X,y

def splitting_data(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 999)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def creating_trees(X_train,y_train,num_of_trees,max_depth):
    trees =[]

    for i in range(num_of_trees):
        tree = DecisionTreeRegressor(max_depth=max_depth,random_state=i)
        tree.fit(X_train,y_train)
        trees.append(tree)
    return trees

def prediction_from_trees(trees,X_test):
    predictions = []
    for tree in trees:
        pred = tree.predict(X_test)
        predictions.append(pred)
    return predictions

def aggregating_predictions(predictions):
    sum=0
    for i in predictions:
        sum=sum+i

    y_pred=sum/len(predictions)
    return y_pred

def evaluating_model(y_test,y_pred):
    r2 = r2_score(y_test,y_pred)
    return r2

def main():
    X,y = data()
    X_train, X_test, y_train, y_test = splitting_data(X,y)
    trees = creating_trees(X_train,y_train,100,2)
    predictions = prediction_from_trees(trees,X_test)
    y_pred = aggregating_predictions(predictions)
    r2 = evaluating_model(y_test,y_pred)
    print("r2_score",r2)



if __name__=="__main__":
    main()




