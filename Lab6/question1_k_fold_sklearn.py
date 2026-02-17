from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
from random import randint
from sklearn.linear_model import LinearRegression
from Lab1.question7 import X
from linear_regression import linear_regression
from sklearn.model_selection import KFold
import pandas as pd

def load_data(filename):
    df = pd.read_csv(filename)
    X=df.iloc[:,0:5].values
    y=df.iloc[:,-1].values
    return X,y


def k_fold_cross_validation(X,y,n_folds=10):
    model=LinearRegression()
    kf=KFold(n_splits=n_folds,shuffle=True,random_state=999)
    scores=[]
    fold=1
    for train_index , test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train,y_train)

        score = model.score(X_test,y_test)
        print(f"Fold {fold}: R² score = {score:.4f}")
        scores.append(score)
        fold+=1

    sum=0
    for i in scores:
        sum+=i
    avg_score=sum/n_folds
    return avg_score

def main():
    X,y=load_data("simulated_data_multiple_linear_regression_for_ML.csv")
    avg_score=k_fold_cross_validation(X,y)
    print(avg_score)

if __name__=="__main__":
    main()

"""Please note that the value of R2-score avg varies in Scratch and sklearn bcoz in the sklearn approach we have used 
randomness or shuffling which we haven't taken in Scratch"""