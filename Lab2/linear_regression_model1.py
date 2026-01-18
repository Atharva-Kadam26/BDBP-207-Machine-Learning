import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier

# def eda(X_df,X_df_scaled):
def main():

#load data
 [X,y]=fetch_california_housing(return_X_y=True)
 print(X.shape)
 print(y.shape)
 # print(X)
 # print(y)

#splitting the data into training set and test set
 [X_train,X_test,y_train,y_test]=train_test_split(X,y,test_size=0.30,random_state=999)
 print(X_train.shape)
 print(y_train.shape)
 print(X_test.shape)
 print(y_test.shape)

#standardizing the data
 scaler=StandardScaler()   #mean and std deviation
 scaler=scaler.fit(X_train)
 X_scale=scaler.transform(X_train)
 X_test_scale=scaler.transform(X_test)

 print(X_scale.shape)
 print(X_test_scale.shape)

#introducing the model
 model=LinearRegression()

#training the model
 model.fit(X_train,y_train)

#predicting the y for test set
 y_pred=model.predict(X_test)
 print(y_pred)


#evaluating the model with r2 score
 r2=r2_score(y_test,y_pred)
 print(r2)


if __name__ == '__main__':
    main()
