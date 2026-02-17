"""K-fold cross validation. Implement for K = 10. Implement from scratch, then, use
scikit-learn methods."""
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
from random import randint
from linear_regression import linear_regression
import pandas as pd

def load_data(filename):
    df = pd.read_csv(filename)
    X_f=df.iloc[:,0:5]
    y_f=df.iloc[:,-1]
    X=X_f.values.tolist()
    y=y_f.values.tolist()
    return X,y

def separating_data_into_folds(k,X,y):
    fold_data=[]
    fold_size=len(X)//k
    for i in range(k):
        X_fold=[]
        y_fold=[]

        start_p = i * fold_size   #eg for i=0 start=0 * 6=0 and end =start + fold_size =0+6=6
        end_p = start_p + fold_size

        for j in range(start_p,end_p):
            X_fold.append(X[j])
            y_fold.append(y[j])

        fold_data.append((X_fold,y_fold))
    return fold_data

def cross_validation(k,X,y,fold_data):
    X_test=[]
    y_test=[]
    X_train = []
    y_train = []
    for i in range(k):
        X_test.append(fold_data[i][0])
        y_test.append(fold_data[i][1])

        for j in range(k):
            if j!=i:
                X_train.extend(fold_data[j][0])
                y_train.extend(fold_data[j][1])

    return X_train,y_train,X_test,y_test

def k_cross_val(k,X,y):
        fold_size=len(X)//k
        X_fold=[]
        y_fold=[]
        for i in range(k):
            start= i * fold_size   #i*6  0*6=0, 1*6=6
            end = start + fold_size  #i+6 0+6=6, 6+6=12

            X_f=X[start:end]
            y_f=y[start:end]

            X_fold.append(X_f)
            y_fold.append(y_f)
        return X_fold,y_fold

def implement_linear_regression(k,X_fold,y_fold):
    r2_scores=[]
    for i in range(k):
        X_test_fold=X_fold[i]
        y_test_fold=y_fold[i]

        X_train_folds=[]
        y_train_folds=[]

        for j in range(k):
            if j!=i:
                X_train_folds.extend(X_fold[j])
                y_train_folds.extend(y_fold[j])

        # X_combined=X_train_folds+ X_test_fold
        # y_combined=y_train_folds + y_test_fold

        r2=linear_regression(X_train_folds,y_train_folds,X_test_fold,y_test_fold,0.01,200,0.1)
        r2_scores.append(r2)

    return r2_scores

def main():
    X, y = load_data('simulated_data_multiple_linear_regression_for_ML.csv')
    data_folds=separating_data_into_folds(10,X,y)
    # print(data_folds)
    df=pd.DataFrame(data_folds)
    # print(df)
    # print(df[0][0])
    # lr=linear_regression.linear_regression('simulated_data_multiple_linear_regression_for_ML.csv',0,5,6,1e-6,1000)
    # print(lr)
    # X_train,y_train,X_test,y_test=cross_validation(10,X,y,data_folds)
    X_fold,y_fold=k_cross_val(10,X,y)
    r2=implement_linear_regression(10,X_fold,y_fold)
    for i in range(len(r2)):
        print("R2 score for fold ",i+1,": ",r2[i])
    sum=0
    for i in r2:
        sum=sum+i
    print(sum/len(r2))











if __name__ == "__main__":
    main()

