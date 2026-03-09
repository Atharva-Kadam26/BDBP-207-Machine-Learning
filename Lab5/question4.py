"""4.Implement logistic regression using scikit-learn for the breast cancer dataset -"""
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

def load_data(filename):
    df=pd.read_csv(filename)
    X_2=df.drop(["diagnosis"],axis=1)
    X=X_2.iloc[:,0:-1]
    return X,df

def one_hot_encoding(df):
    df['diagnosis']=df['diagnosis'].map({'M':1,'B':0})
    y=df['diagnosis']
    return y

def splitting_data(X,y):
    X_tr, X_te, y_tr, y_te = train_test_split(X,y,test_size=0.3,random_state=999)
    return X_tr, X_te, y_tr, y_te

# def remove_null_values(X_tr,X_te):
#     inputer=SimpleImputer(strategy='mean')
#     X_train=inputer.fit_transform(X_tr)
#     X_test=inputer.transform(X_te)
#     return X_train,X_test

def standardizing_data(X_train,X_test):
    scaler = StandardScaler()
    X_train_scale=scaler.fit_transform(X_train)
    X_test_scale=scaler.transform(X_test)
    return X_train_scale,X_test_scale

def logistic_regression(X_train_scale,y_tr,X_test_scale):
    model=LogisticRegression(max_iter=2000)
    model.fit(X_train_scale,y_tr)
    y_pred=model.predict(X_test_scale)
    return y_pred

def accuracy(y_test,y_pred):
    acc=accuracy_score(y_test,y_pred)
    return acc

def main():
    X,df=load_data("breast_cancer_dataset.csv")
    y=one_hot_encoding(df)
    X_train, X_test, y_train, y_test = splitting_data(X,y)
    # X_train,X_test=remove_null_values(X_tr,X_te)
    X_train_scaled,X_test_scaled=standardizing_data(X_train,X_test)
    y_pred=logistic_regression(X_train_scaled,y_train,X_test_scaled)
    print(y_pred)
    print(y_test.tolist())
    acc=accuracy(y_test,y_pred)
    print(acc)





if __name__ == "__main__":
    main()


