"""Data standardization - scale the values such that mean of new dist = 0 and sd = 1.
Implement code from scratch."""
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import StandardScaler

def load_data(filename):
    df = pd.read_csv(filename)
    X_f = df.iloc[:, 0:5]
    y_f = df.iloc[:, -1]
    X = X_f.values.tolist()
    y = y_f.values.tolist()
    return X, y

def standardize(X):
    """Standardizing with Z-score
    x_new= x-mean/std deviation"""

    """Mean calculation"""
    mean_data=[]
    columns=len(X[0])
    rows=len(X)

    for i in range(columns):
        col=[]
        for j in range(rows):
            col.append(X[j][i])
        mean=sum(col)/len(col)
        mean_data.append(mean)

    """Standard Deviation"""
    std_dev=[]
    columns=len(X[0])
    rows=len(X)
    for i in range(columns):
        col_feature=[]
        for j in range(rows):
            col_feature.append(X[j][i])

        std=0
        for k in range(len(col_feature)):
            std+=((col_feature[k]-mean_data[i])**2)/len(col_feature)
        std_dev.append(math.sqrt(std))


    """Standardizing with Z-score"""
    standardized_X = []
    columns=len(X[0])
    rows=len(X)
    for i in range(rows):
        new_row=[]
        for j in range(columns):
            x_new=(X[i][j]-mean_data[j])/std_dev[j]
            new_row.append(x_new)
        standardized_X.append(new_row)
    return standardized_X

def verifying_mean_Xnew(standardized_X):
    mean_new_data=[]
    cols=len(standardized_X[0])
    rows=len(standardized_X)
    for i in range(cols):
        sum_num=0
        avg=0
        for j in range(rows):
            sum_num+=standardized_X[j][i]
            avg=round(sum_num/len(standardized_X))
        mean_new_data.append(avg)
    return mean_new_data

def verifying_variance_Xnew(standardized_X,mean_new_data):
    variance_new_X=[]
    columns = len(standardized_X[0])
    rows = len(standardized_X)
    for i in range(columns):
        col_feature = []
        for j in range(rows):
            col_feature.append(standardized_X[j][i])

        std = 0
        for k in range(len(col_feature)):
            std += ((col_feature[k] - mean_new_data[i]) ** 2) / len(col_feature)
        variance_new_X.append(round(math.sqrt(std))**2)
    return variance_new_X



def scaling_w_sklearn(X):
    scaler=StandardScaler()
    X_scaled=scaler.fit_transform(X)
    return X_scaled

def main():
    X, y = load_data("simulated_data_multiple_linear_regression_for_ML.csv")

    m=standardize(X)
    print("Scaled values from scratch are:",m)

    X_scaled=scaling_w_sklearn(m)
    print("Scaled values using sklearn are:", X_scaled)

    mean=verifying_mean_Xnew(m)
    print("The mean of X-new values are:",mean)

    variance=verifying_variance_Xnew(m,mean)
    print("The variance of X-new values are:",variance)


if __name__=="__main__":
    main()