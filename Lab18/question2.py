"""Try classifying classes 1 and 2 from the iris dataset with SVMs, with the 2 first features.
Leave out 10% of each class and test prediction performance on these observations.
https://scikit-learn.org/stable/tutorial/statistical_inference/supervised_learning.html#super
vised-learning-tut - Check the solution code to learn about various plots."""
from pandas.core.config_init import pc_nb_repr_h_doc
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

def load_data():
    iris = load_iris()
    #As only two classes are asked for
    X = iris.data[iris.target != 0, 2:4] # removing class 0: setosa , 2 and 3 are the column selected i.e the petal features
    y = iris.target[iris.target != 0]
    return X, y

def split(X, y):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=999)
    return X_train, X_test, y_train, y_test

def svm(X_train,y_train):
    model = SVC(kernel='rbf')
    trained_model=model.fit(X_train,y_train)
    return trained_model

def predict(model,X_test):
    y_pred = model.predict(X_test)
    return y_pred

def score(y_test,prediction):
    acc_score = accuracy_score(y_test,prediction)
    return acc_score

def plotting(model,X,y):
    X_min,X_max = X[:,0].min() -1 ,X[:,0].max() +1
    y_min,y_max = X[:,1].min() -1 ,X[:,1].max() +1

    xx,yy = np.meshgrid(
        np.linspace(X_min,X_max,100),
        np.linspace(y_min,y_max,100)
    )

    #Prepare grid points
    grid = np.column_stack([xx.reshape(-1),yy.reshape(-1)]) #flattening the matrix

    Z= model.predict(grid)
    Z = Z.reshape(xx.shape)

    #Plot decision regions
    plt.contourf(xx,yy,Z,alpha=0.3)

    #Plot actual points
    plt.scatter(X[:,0],X[:,1],c=y,edgecolors='k')

    plt.xlabel("feature 1")
    plt.ylabel("feature 2")
    return plt.show()


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split(X, y)
    trained_model = svm(X_train, y_train)
    y_pred = predict(trained_model,X_test)
    print("The predicted values are:",y_pred)
    acc_score = score(y_test,y_pred)
    print("The accuracy score is:",acc_score)
    graph= plotting(trained_model,X_test,y_test)
    print(graph)




if __name__ == "__main__":
    main()
