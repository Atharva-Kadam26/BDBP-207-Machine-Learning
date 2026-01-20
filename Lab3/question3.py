"""Use the above simulated CSV file and implement the following from scratch in Python"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(filename):
    simulated_data=pd.read_csv(filename)
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

def test_function(X_train_scaled_up,theta,y_train):
    squared_error=0

    h0=hypo_func(X_train_scaled_up,theta)

    for i in range(len(y_train)):
        y_train_list=y_train.tolist()
        error = h0[i] - y_train_list[i]
        squared_error = squared_error + error**2
    total_err=squared_error/2
    return total_err

def derivative(theta,X_train_scaled_up,y_train,alpha=0.01):
    y_train_list=y_train.tolist()
    new_thetas=theta.copy()  #for old thetas
    old_thetas=theta.copy()
    exp_val=hypo_func(X_train_scaled_up,theta)

    for i in range(len(theta)):      #loop over theta
        gradient=0             #summation(h-y)xj
        for j in range(len(y_train_list)):   #all rows
            err= exp_val[j] - y_train_list[j]
            gradient += err * X_train_scaled_up[j][i]  #xj(i)
        new_thetas[i] = theta[i] - alpha * gradient
    return new_thetas,old_thetas

def convergence(old_thetas,new_thetas,tol=1e-6):
    for i in range(len(old_thetas)):
        print("Old Thetas are:",old_thetas[i],"New Thetas are:",new_thetas[i])
        if abs(new_thetas[i]-old_thetas[i]) > tol:
            return False
    return True

def grad_desc_until_convergence(X_train_scaled_up,y_train,alpha=0.01,tol=1e-6, iterations=10000):
    theta=[0]*X_train_scaled_up.shape[1]

    for i in range(iterations):
        new_thetas,old_thetas=derivative(theta,X_train_scaled_up,y_train,alpha)
        print(f"\nInteration {i +1}")
        print("OLd Theta=",old_thetas)
        print("New Theta=",new_thetas)

        if convergence(old_thetas,new_thetas,tol):
            print("Converged")
            break

        theta=new_thetas
    return theta

def y_pred(X_test_scaled,grad):
    x0=np.ones((X_test_scaled.shape[0],1))
    X_test_scaled_up=np.hstack((x0,X_test_scaled))
    y_predicted=hypo_func(X_test_scaled_up,grad)
    return y_predicted

def r2_score(y_test,y_predicted):
    y_test_list=y_test.tolist()
    y_test_mean=sum(y_test_list)/len(y_test_list)

    sum_of_square_t=0
    for i in range(len(y_test_list)):   #total sum
        sum_of_square_t +=((y_test_list[i] - y_test_mean)**2)

    sum_of_square_r=0
    for i in range(len(y_test_list)):   #residual sum
        sum_of_square_r+=((y_test_list[i] - y_predicted[i])**2)


    r2= 1 - (sum_of_square_r/sum_of_square_t)
    return r2


def main():
    X,y=load_data("simulated_data_multiple_linear_regression_for_ML.csv")
    X_train,X_test,y_train,y_test=split_data(X,y)
    X_train_scaled,X_test_scaled=standardize_data(X_train,X_test)
    X_train_scaled_up=x0_intercept(X_train_scaled)
    theta=init_theta(X_train_scaled_up)
    h=hypo_func(X_train_scaled_up,theta)
    J0=test_function(X_train_scaled_up,theta,y_train)
    new_thetas,old_thetas=derivative(theta,X_train_scaled_up,y_train)
    conv=convergence(old_thetas,new_thetas,tol=1e-8)
    grad=grad_desc_until_convergence(X_train_scaled_up,y_train)
    y_predicted=y_pred(X_test_scaled,grad)
    print("y-predicted values:",y_predicted)
    r2=r2_score(y_test,y_predicted)
    print("R2 score:",r2)

    return h,J0,conv,grad,y_predicted


if __name__=="__main__":
    main()

