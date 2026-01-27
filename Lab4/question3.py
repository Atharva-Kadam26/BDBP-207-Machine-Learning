"""Implement normal equations method from scratch and compare your results on a
simulated dataset (disease score fluctuation as target) and the admissions dataset
(https://www.kaggle.com/code/erkanhatipoglu/linear-regression-using-the-normal-equatio
n ). You can compare the results with scikit-learn and your own gradient descent
implementation"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

# def load_data(filename):
#     data = pd.read_csv(filename)
#     X = data.iloc[:,0:5]
#     y = data.iloc[:,6]
#     return X,y
#
# def split_data(X,y):
#     X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=999)
#     return X_train,X_test,y_train,y_test
#
# def standardize_data(X_train,X_test):
#     scaler = StandardScaler()
#     X_train_scale= scaler.fit_transform(X_train)
#     X_test_scale=scaler.transform(X_test)
#     return X_train_scale,X_test_scale
#
# def
#
#
# def main():
#     X,y= load_data("/home/ibab/machine_learning/Lab3/simulated_data_multiple_linear_regression_for_ML.csv")
#     X_train, X_test, y_train, y_test = split_data(X,y)
#     X_train_scale,X_test_scale= standardize_data(X_train,X_test)
#
#
#
#
#
# if __name__ == "__main__":
#     main()
data= [
    ["x1","x2","y"],
    [1,2,5],
    [3,4,11]
]
#theta=(Xt.X)^-1.Xt.y
def load_data(dataset):
    X=[]
    y=[]
    for row in range(1,len(dataset)):
        X.append(dataset[row][:-1])
        y.append(dataset[row][-1])
    return X,y
        # for col in range(len(dataset)):
        #     X.append(dataset[row][col])

def X_transpose(X):
    X_transpose=[]
    for i in range(len(X[0])):
        row=[]
        for j in range(len(X)):
            row.append(X[j][i])
        X_transpose.append(row)
    return X_transpose

def X_t_X(X_transpose,X):
    XT_X=[]
    for i in range(len(X_transpose)):
        row=[]
        for j in range(len(X[0])):
            product=0
            for k in range(len(X)):
                product+=X_transpose[i][k]* X[k][j]
            row.append(product)
        XT_X.append(row)
    return XT_X

def Xt_X_inverse(XT_X):
    XT_X_inv=np.linalg.inv(XT_X)
    return XT_X_inv

def Xt_X_y(Xt_X,y):
    XT_X_y=[]
    for i in range(len(Xt_X)):
        product=0
        for j in range(len(Xt_X[0])):
            product+=Xt_X[j][i]* y[j]
        XT_X_y.append(product)
    return XT_X_y

def finale(XT_X_y,XT_X_inv):
    finale=[]
    for i in range(len(XT_X_inv)):
        product=0
        for j in range(len(XT_X_inv[0])):
            product+=XT_X_inv[j][i]* XT_X_y[j]
        finale.append(float(product))
    return finale

def main():
    X,y=load_data(data)
    X_t=X_transpose(X)
    print("X=",X)
    print("y=",y)
    print("X_t=",X_t)
    XtX=X_t_X(X_t,X)
    print("Xt_X=",XtX)
    det_X=Xt_X_inverse(X)
    Xt_y=Xt_X_y(X,y)
    print("Xt_y=",Xt_y)
    final=finale(Xt_y,Xt_X_inverse(XtX))
    print(final)


if __name__=="__main__":
    main()
