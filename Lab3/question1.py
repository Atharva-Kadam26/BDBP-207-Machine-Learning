
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
def main():
 df=pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
 print(df.head())

 #removing the column disease_score_fluct
 df_upd=df.drop("disease_score_fluct",axis=1)
 print(df_upd.head())

#loading the data
 X=df_upd.drop("disease_score",axis=1)   #using indices=> X=df.iloc[:,0:5]  & y=df.iloc[:,6]
 y=df_upd["disease_score"]
 # print(X.shape)
 # print(y.shape)

 #splitting the data into training and test set
 [X_train, X_test, y_train, y_test]=train_test_split(X,y,test_size=0.3,random_state=999)
 # print(X_train.shape)
 # print(y_train.shape)
 # print(X_test.shape)
 # print(y_test.shape)

 #standardizing the data
 scaler=StandardScaler()
 scaler=scaler.fit(X_train)
 X_scale=scaler.transform(X_train)
 X_test_scale=scaler.transform(X_test)


 #introducing the model
 model=LinearRegression()

 #training the model
 model.fit(X_scale,y_train)

 #predicting the y for test set
 y_pred=model.predict(X_test_scale)
 print(y_pred)

 #evaluating the model with r2 score
 r2=r2_score(y_test,y_pred)
 print(r2)

if __name__ == '__main__':
    main()


