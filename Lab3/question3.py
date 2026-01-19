"""Use the above simulated CSV file and implement the following from scratch in Python"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(filename):
 simulated_data=pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
 X=simulated_data.iloc[:,0:5]
 y=simulated_data.iloc[:,6]
 return X,y

def split_data(X,y):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
    return X_train,X_test,y_train,y_test

def standardize_data(X_train,X_test):
    scaler=StandardScaler()
    X_train_scaled=scaler.fit_transform(X_train)
    X_test_scaled=scaler.transform(X_test)
    return X_train_scaled,X_test_scaled

def x0_intercept(X_train_scaled):
    x0=np.ones((X_train_scaled.shape[0],1))
    X_train_scaled_up=np.hstack((x0,X_train_scaled))
    return X_train_scaled_up

def init_theta(X_train_scaled_up):
    theta=[0]*X_train_scaled_up.shape[1]        #shape[1] = no. of columns
    return theta

def hypo_func(X_train_scaled_up,theta):
    exp_val=[]                                  #exp_val=h0(x)
    for i in range(len(X_train_scaled_up)):    #features xi
        x_row=X_train_scaled_up[i]
        h0=0

        for j in range(len(theta)):            #thetas or parameters
            h0 += theta[j] * x_row[j]
        exp_val.append(h0)
    return exp_val

def test_function(X_train_scaled_up,new_theta,y_train):
    squared_error=0
    for i in range(len(y_train)):
        squared_error += (new_theta[i] - y_train[i])**2   #squared error





def main():
    X,y=load_data("simulated_data_multiple_linear_regression_for_ML.csv")
    X_train,X_test,y_train,y_test=split_data(X,y)
    X_train_scaled,X_test_scaled=standardize_data(X_train,X_test)
    X_train_scaled_up=x0_intercept(X_train_scaled)
    theta=init_theta(X_train_scaled_up)
    h=hypo_func(X_train_scaled_up,theta)
    return h

print(main())

if __name__=="__main__":
    main()




# def hypothesis_func(X_train_scaled,thetas):
#     hyp=0
#     for i in range(1,len(X_train_scaled)+1):
#         hyp += thetas[i] * X_train_scaled[i]   #hyp_func=theta0X0 + theta1X1 + thetadXd
#     return hyp
