"""1.Implement Stochastic Gradient Descent algorithm from scratch"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from random import randint
from sklearn.datasets import fetch_california_housing


def load_data(filename):
    # data= pd.read_csv(filename)
    [X_df,y_df]=fetch_california_housing(return_X_y=True)
    X = X_df.tolist()
    y = y_df.tolist()
    # X = X_d.iloc[:, 0:5]
    # y = X_d.iloc[:, 6]
    return X,y


def split_data(X,y):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=999)
    return X_train,X_test,y_train,y_test

def standardize_data(X_train,X_test):
    scaler= StandardScaler()
    X_train_scaled= scaler.fit_transform(X_train)
    X_test_scaled=scaler.transform(X_test)
    return X_train_scaled,X_test_scaled

def x0_intercept(X_train_scaled):
    x0=np.ones((X_train_scaled.shape[0],1))
    X_train_scale_int=np.hstack((x0,X_train_scaled))
    len_X = len(X_train_scale_int[0])
    return X_train_scale_int,len_X

def initialise_thetas(len_X):
    thetas=[]
    for i in range(len_X):
        thetas.append(0)
    return thetas

def hypothesis_func(X_train_scale_int,thetas):
    predictions=[]
    for i in range(len(X_train_scale_int)):
        x_row=X_train_scale_int[i]
        h=0
        for j in range(len(x_row)):
            h+=thetas[j] * float(x_row[j])  #hθ(x)=θ0x0+θ1x1+⋯+θnxn i.e.h=θ(j)x(j)^(i)  where i is sample number
        predictions.append(h)

    return predictions

def cost_func(X_train_scale_int,thetas,y_train,predictions):
    total_error=0
    for i in range(len(y_train)):
        total_error+=(predictions[i] - y_train[i])**2
    cost=total_error/2
    return cost

def stochastic_gradient_descent(X_train_scale_int,thetas,y_train,alpha=1e-6,iterations=50,tolerance=1e-6):
    cost_func_h=[]
    for iteration in range(iterations):
        old_thetas=thetas.copy()
        # pred=hypothesis_func(X_train_scale_int,thetas)

        for _ in range(len(y_train)):
            i=randint(0,len(y_train)-1)
            x_rand=X_train_scale_int[i]
            y_rand=y_train[i]

            h=0
            for j in range(len(thetas)):
                h+= float(thetas[j] * x_rand[j])

            for j in range(len(thetas)):
                gradient=(h -y_rand) *x_rand[j]
                thetas[j]-= alpha * gradient

        theta_change=0
        for j in range(len(thetas)):
            theta_change+=abs(thetas[j] - old_thetas[j])

        if theta_change < tolerance:
            print(f"Convergence at iteration {iteration}")
            break

        # print(f"Iteration {iteration+1} | Theta change = {theta_change} ")
    return thetas

def predict_y_test(X_test_scaled,thetas):
    x0=np.ones((X_test_scaled.shape[0],1))
    X_test_scale_int=np.hstack((x0,X_test_scaled))

    y_test_pred=[]
    for i in range(len(X_test_scale_int)):
        X_test_row=X_test_scale_int[i]
        h=0
        for j in range(len(thetas)):
            h+= float(thetas[j] * X_test_row[j])
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

def main():
    X,y=load_data("")
    X_train, X_test, y_train, y_test = split_data(X,y)
    X_train_scaled,X_test_scaled=standardize_data(X_train,X_test)
    X_train_scale_int,len_X=x0_intercept(X_train_scaled)
    thetas=initialise_thetas(len_X)
    predictions=hypothesis_func(X_train_scale_int,thetas)
    total_error=cost_func(X_train_scale_int,thetas,y_train,predictions)
    thetas=stochastic_gradient_descent(X_train_scale_int,thetas,y_train,alpha=1e-6,iterations=1000)
    # print("Final Thetas:",thetas)
    # print("Total error =",total_error)
    y_test_pred=predict_y_test(X_test_scaled,thetas)
    # print("Y_test_pred:",y_test_pred)
    # print(len(y_test_pred))
    # print("Y_test:",y_test)
    # print(len(y_test))
    r2=r2_score(y_test,y_test_pred)
    print("R2 Score=",r2)

    # # print(cost)
    # print(predictions)


if __name__ == "__main__":
    main()
