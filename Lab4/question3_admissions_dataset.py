#Using own Gradient Descent
#Admissions Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

def load_data(filename):
    data = pd.read_csv(filename)
    X = data.iloc[:,0:8]
    y = data.iloc[:,8]
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

# def r2_score(y_test,y_test_pred):
#     y_test_list=y_test
#     y_test_mean=sum(y_test_list)/len(y_test_list)
#
#     sum_of_square_t=0
#     for i in range(len(y_test_list)):   #total sum
#         sum_of_square_t +=((y_test_list[i] - y_test_mean)**2)
#
#     sum_of_square_r=0
#     for i in range(len(y_test_list)):   #residual sum
#         sum_of_square_r+=((y_test_list[i] - y_test_pred[i])**2)
#
#     r2= 1 - (sum_of_square_r/sum_of_square_t)
#     return r2

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
    X,y=load_data("Admission_Predict.csv")
    X_train, X_test, y_train, y_test_l= split_data(X,y)
    X_train_scale,X_test_scale= standardize_data(X_train,X_test)
    X_train_scale,X_test_scale= intercept(X_train_scale,X_test_scale)
    X_t=X_transpose(X_train_scale)
    Xt_X =X_t_X(X_train_scale,X_t)
    XtX_inv=Xt_X_inv(Xt_X)
    Xt_y= Xty(X_t,y_train)
    thetas=theta(XtX_inv,Xt_y)
    y_test_pred=predict_y_test(X_test_scale,thetas)
    # print("y test values: ",y_test_l)
    # print("y predicted values: ",y_test_pred)
    r2=r2_score(y_test_pred,y_test_l)
    print("R2 score by Normal Equation is:",r2)



if __name__ == '__main__':
    main()


#By using Scikit-learn model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
def load_data(filename):
    df=pd.read_csv(filename)
    X=df.iloc[:,0:8]
    y=df.iloc[:,8]
    return X,y

def split_data(X,y):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=999)
    return X_train,X_test,y_train,y_test

def standardize_data(X_train,X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scale=scaler.transform(X_train)
    X_test_scale=scaler.transform(X_test)
    return X_train_scale,X_test_scale

def linear_regression(X_train_scale,y_train):
    model=LinearRegression()
    model.fit(X_train_scale,y_train)
    return model

def predicting_y(model,X_test_scale):
    y_pred=model.predict(X_test_scale)
    return y_pred

def evaluating(y_test,y_pred):
    r2=r2_score(y_test,y_pred)
    return r2

def main():
    X,y=load_data("Admission_Predict.csv")
    X_train,X_test,y_train,y_test=split_data(X,y)
    X_train_scale,X_test_scale=standardize_data(X_train,X_test)
    model=linear_regression(X_train_scale,y_train)
    y_pred=predicting_y(model,X_test_scale)
    r2=evaluating(y_test,y_pred)
    print("R2 score by Scikit-Learn Model is:",r2)
    return "Scikit-learn model executed successfully"


if __name__=="__main__":
    main()


#Implementing Gradient Gradient on Admissions.csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_data(filename):
    data= pd.read_csv(filename)
    X_df = data.iloc[:,0:8]
    y_df = data.iloc[:,8]
    X = X_df.values.tolist()
    y = y_df.values.tolist()
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

# def cost_func(X_train_scale_int,thetas,y_train,predictions):
#     total_error=0
#     for i in range(len(y_train)):
#         total_error+=(predictions[i] - y_train[i])**2
#     cost=total_error/2
#     return cost

def gradient_descent(X_train_scale_int,thetas,y_train,alpha,iterations,tolerance=1e-6):
    cost_func_h=[]
    for iteration in range(iterations):
        old_thetas=thetas.copy()
        pred=hypothesis_func(X_train_scale_int,thetas)

        for j in range(len(thetas)):
            gradient=0
            for i in range(len(y_train)):
                gradient+=(pred[i] - y_train[i]) *X_train_scale_int[i][j]
            thetas[j] = float(thetas[j] - alpha * gradient)

        theta_change=0
        for j in range(len(thetas)):
            theta_change+=abs(thetas[j] - old_thetas[j])

        if theta_change < tolerance:
            # print(f"Convergence at iteration {iteration}")
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
    X,y=load_data("Admission_Predict.csv")
    X_train, X_test, y_train, y_test = split_data(X,y)
    X_train_scaled,X_test_scaled=standardize_data(X_train,X_test)
    X_train_scale_int,len_X=x0_intercept(X_train_scaled)
    thetas=initialise_thetas(len_X)
    predictions=hypothesis_func(X_train_scale_int,thetas)
    # total_error=cost_func(X_train_scale_int,thetas,y_train,predictions)
    thetas=gradient_descent(X_train_scale_int,thetas,y_train,alpha=1e-4,iterations=5000)
    y_test_pred=predict_y_test(X_test_scaled,thetas)
    r2=r2_score(y_test,y_test_pred)
    print("R2 Score by Gradient Descent from scratch=",r2)
    return "y values predicted by Gradient Descent successfully."

if __name__ == "__main__":
    main()


