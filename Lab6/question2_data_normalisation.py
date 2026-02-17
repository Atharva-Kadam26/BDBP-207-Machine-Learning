"""Data normalization - scale the values between 0 and 1. Implement code from scratch."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data(filename):
        df = pd.read_csv(filename)
        X_f = df.iloc[:, 0:5]
        y_f = df.iloc[:, -1]
        X=X_f.values.tolist()
        y=y_f.values.tolist()
        return X, y

def minim_maxim(X):
    cols=len(X[0])
    rows=len(X)

    minim=[]
    maxim=[]
    for i in range(cols):
        column=[]
        for j in range(rows):
            column.append(X[j][i])
        minim.append(min(column))
        maxim.append(max(column))

    return minim,maxim

def normalize(minim,maxim,X):
    norm=[]
    cols=len(X[0])
    rows=len(X)
    for i in range(cols):
        row=[]
        for j in range(rows):
            x_new=(X[j][i]-minim[i])/(maxim[i]-minim[i])
            row.append(x_new)
        norm.append(row)
    return norm


def main():
    X,y = load_data("simulated_data_multiple_linear_regression_for_ML.csv")
    minim,maxim=minim_maxim(X)
    norm=normalize(minim,maxim,X)
    print(norm)

    scaler = MinMaxScaler(feature_range=(0, 1))
    X_s = scaler.fit_transform(X)
    print(X_s)



if __name__ == "__main__":
    main()
