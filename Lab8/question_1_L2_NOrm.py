"""Implement L2-norm and L1-norm from scratch"""


"""L2-Norm From Scratch"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_data(filename):
    data= pd.read_csv(filename)
    X_df = data.iloc[:,0:5]
    y_df = data.iloc[:,6]
    X = X_df.values.tolist()
    y = y_df.values.tolist()
    return X,y


def split_data(X,y):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=999)
    y_train=np.array(y_train)
    y_test=np.array(y_test)
    return X_train,X_test,y_train,y_test

def standardize_data(X_train,X_test):
    scaler= StandardScaler()
    X_train_scaled= scaler.fit_transform(X_train)
    X_test_scaled=scaler.transform(X_test)

    X_test_scaled=np.array(X_test_scaled)
    return X_train_scaled,X_test_scaled

def x0_intercept(X_train_scaled):
    x0=np.ones((X_train_scaled.shape[0],1))
    X_train_scale_int=np.hstack((x0,X_train_scaled))
    len_X = len(X_train_scale_int[0])
    x_train = np.array(X_train_scale_int)
    return x_train,len_X

def initialise_thetas(len_X):
    thetas=[]
    for i in range(len_X):
        thetas.append(0)
    theta = np.array(thetas,dtype=float)
    return theta

def hypothesis_func(x_train,theta):
    h_0x=[]

    for i in range(len(x_train)):
        total=0
        for j in range(len(theta)):
            total += theta[j] * x_train[i][j]
        h_0x.append(total)
    return np.array(h_0x)

def cost_function(x_train,y_train,theta,lamb):
    total_error=0
    predictions = hypothesis_func(x_train,theta)

    for i in range(len(y_train)):
        total_error += (y_train[i]-predictions[i])**2


    """L1 Regularisation"""
    l1_norm = 0
    for j in range(1,len(theta)):
        l1_norm += theta[j]

    return total_error + lamb * l1_norm

def gradient_descent_with_l1_norm(x_train,y_train,theta,alpha,lamb,iterations):

    for i in range(iterations):
        predictions = hypothesis_func(x_train,theta)
        error = predictions - y_train   #J_0x=(h_0x - y)

        for j in range(len(theta)):
            gradient =0

            for k in range(len(y_train)):
                gradient += error[k] * x_train[k][j]

            #L2 term
            if j==0:
                reg_term = 0
            else:
                reg = 2*(lamb * theta[j])

                theta[j] = theta[j] - alpha * (gradient + reg)

    return theta

    total_error=0
def main():
    X,y = load_data("simulated_data_multiple_linear_regression_for_ML.csv")
    X_train, X_test, y_train, y_test = split_data(X,y)
    X_train_scaled, X_test_scaled = standardize_data(X_train,X_test)
    x_train, len_X = x0_intercept(X_train_scaled)
    theta= initialise_thetas(len_X)
    h_0x=hypothesis_func(x_train,theta)
    j_0x=cost_function(x_train,y_train,theta,0.1)
    up_thetas=gradient_descent_with_l1_norm(x_train,y_train,theta,0.001,0.1,2000)
    print(up_thetas)


if __name__=="__main__":
    main()
