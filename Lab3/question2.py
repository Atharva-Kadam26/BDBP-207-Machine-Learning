from fileinput import filename

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
def load_data(filename):
    df=pd.read_csv(filename)
    X=df.iloc[:,0:5]
    y=df.iloc[:,6]
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
    X,y=load_data("simulated_data_multiple_linear_regression_for_ML.csv")
    X_train,X_test,y_train,y_test=split_data(X,y)
    X_train_scale,X_test_scale=standardize_data(X_train,X_test)
    model=linear_regression(X_train_scale,y_train)
    y_pred=predicting_y(model,X_test_scale)
    r2=evaluating(y_test,y_pred)
    return r2

output=main()
print("R2 Score is:",output)

if __name__=="__main__":
    main()
