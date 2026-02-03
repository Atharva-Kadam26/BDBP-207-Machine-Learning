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
# data= [
#     ["x1","x2","y"],
#     [1,2,5],
#     [3,4,11]
# ]
# #theta=(Xt.X)^-1.Xt.y
# def load_data(dataset):
#     X=[]
#     y=[]
#     for row in range(1,len(dataset)):
#         X.append(dataset[row][:-1])
#         y.append(dataset[row][-1])
#     return X,y
#         # for col in range(len(dataset)):
#         #     X.append(dataset[row][col])
#
# def X_transpose(X):
#     X_transpose=[]
#     for i in range(len(X[0])):
#         row=[]
#         for j in range(len(X)):
#             row.append(X[j][i])
#         X_transpose.append(row)
#     return X_transpose
#
# def X_t_X(X_transpose,X):
#     XT_X=[]
#     for i in range(len(X_transpose)):
#         row=[]
#         for j in range(len(X[0])):
#             product=0
#             for k in range(len(X)):
#                 product+=X_transpose[i][k]* X[k][j]
#             row.append(product)
#         XT_X.append(row)
#     return XT_X
#
# def Xt_X_inverse(XT_X):
#     XT_X_inv=np.linalg.inv(XT_X)
#     return XT_X_inv
#
# def Xt_X_y(Xt_X,y):
#     XT_X_y=[]
#     for i in range(len(Xt_X)):
#         product=0
#         for j in range(len(Xt_X[0])):
#             product+=Xt_X[j][i]* y[j]
#         XT_X_y.append(product)
#     return XT_X_y
#
# def finale(XT_X_y,XT_X_inv):
#     finale=[]
#     for i in range(len(XT_X_inv)):
#         product=0
#         for j in range(len(XT_X_inv[0])):
#             product+=XT_X_inv[j][i]* XT_X_y[j]
#         finale.append(float(product))
#     return finale
#
# def main():
#     X,y=load_data(data)
#     X_t=X_transpose(X)
#     print("X=",X)
#     print("y=",y)
#     print("X_t=",X_t)
#     XtX=X_t_X(X_t,X)
#     print("Xt_X=",XtX)
#     det_X=Xt_X_inverse(X)
#     Xt_y=Xt_X_y(X,y)
#     print("Xt_y=",Xt_y)
#     final=finale(Xt_y,Xt_X_inverse(XtX))
#     print(final)


# if __name__=="__main__":
#     main()

def load_data(filename):
    data = pd.read_csv(filename)
    X = data.iloc[:,0:5]
    y = data.iloc[:,6]
    return X,y

def split_data(X,y):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=999)
    y_test_l=y_test.tolist()
    return X_train,X_test,y_train,y_test_l

def standardize_data(X_train,X_test):
    scaler = StandardScaler()
    X_train_s= scaler.fit_transform(X_train)
    X_test_s=scaler.transform(X_test)
    return X_train_s,X_test_s

def intercept(X_train_s,X_test_s):
    x0=np.ones((X_train_s.shape[0],1))
    X_train_scale=np.hstack((x0,X_train_s))

    x0 = np.ones((X_test_s.shape[0], 1))
    X_test_scale = np.hstack((x0, X_test_s))

    return X_train_scale,X_test_scale

def X_transpose(X_train_scale):
    X_t=[]
    for i in range(len(X_train_scale[0])):
        row=[]
        for j in range(len(X_train_scale)):
            row.append(X_train_scale[j][i])
        X_t.append(row)
    return X_t

def X_t_X(X_train_scale,X_t):
    Xt_X=[]
    for i in range(len(X_t)):
        row=[]
        for j in range(len(X_train_scale[0])):
            product=0
            for k in range(len(X_train_scale)):
                product+=X_t[i][k] * X_train_scale[k][j]
            row.append(float(product))
        Xt_X.append(row)
    return Xt_X

def Xt_X_inv(Xt_X):
    XtX_inv=np.linalg.inv(Xt_X)
    return XtX_inv

def Xty(X_t,y_train):
    Xt_y=[]
    y_train_l=y_train.tolist()
    for i in range(len(X_t)):
        product=0
        for j in range(len(X_t[0])):
            product+=X_t[i][j]*y_train_l[j]
        Xt_y.append(product)
    return Xt_y

def theta(XtX_inv,Xt_y):
    thetas=[]
    for i in range(len(XtX_inv)):
        product=0
        for j in range(len(XtX_inv[0])):
            product+=XtX_inv[i][j]*Xt_y[j]
        thetas.append(float(product))
    return thetas

def predict_y_test(X_test_scale,thetas):
    y_test_pred=[]
    for i in range(len(X_test_scale)):
        X_test_row=X_test_scale[i]
        h=0
        for j in range(len(thetas)):
            h+=float(thetas[j]*X_test_row[j])
        y_test_pred.append(h)
    return y_test_pred

def r2_score(y_test,y_test_pred):
    y_test_list=y_test
    y_test_mean=sum(y_test_list)/len(y_test_list)

    sum_of_square_t=0
    for i in range(len(y_test_list)):   #total sum
        sum_of_square_t +=((y_test_list[i] - y_test_mean)**2)

    sum_of_square_r=0
    for i in range(len(y_test_list)):   #residual sum
        sum_of_square_r+=((y_test_list[i] - y_test_pred[i])**2)

    r2= 1 - (sum_of_square_r/sum_of_square_t)
    return r2

def r2_score(y_test_pred,y_test_l):
    y_test_mean=sum(y_test_l)/len(y_test_l)
    sum_of_square_t=0
    for i in range(len(y_test_l)):
        sum_of_square_t +=((y_test_l[i] - y_test_mean)**2)

    sum_of_square_r=0
    for i in range(len(y_test_pred)):
        sum_of_square_r+=((y_test_l[i] - y_test_pred[i])**2)

    r2=1 - (sum_of_square_r/sum_of_square_t)
    return r2



def main():
    X,y=load_data("simulated_data_multiple_linear_regression_for_ML.csv")
    X_train, X_test, y_train, y_test_l= split_data(X,y)
    X_train_scale,X_test_scale= standardize_data(X_train,X_test)
    X_train_scale,X_test_scale= intercept(X_train_scale,X_test_scale)
    X_t=X_transpose(X_train_scale)
    Xt_X =X_t_X(X_train_scale,X_t)
    XtX_inv=Xt_X_inv(Xt_X)
    Xt_y= Xty(X_t,y_train)
    thetas=theta(XtX_inv,Xt_y)
    y_test_pred=predict_y_test(X_test_scale,thetas)
    print("y test values: ",y_test_l)
    print("y predicted values: ",y_test_pred)
    r2=r2_score(y_test_pred,y_test_l)
    print("R2 score is:",r2)



if __name__ == '__main__':
    main()

