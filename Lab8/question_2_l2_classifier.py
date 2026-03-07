import pandas as pd
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def load_data(filename):
    df=pd.read_csv(filename)
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    return X,y

def splitting_data(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    return X_train, X_test, y_train, y_test

def scaling_data(X_train,X_test):
    scaler = StandardScaler()
    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)
    return X_train,X_test

def train_model(X_train,y_train):
    clf = RidgeClassifier()
    ridge_mod = clf.fit(X_train, y_train)
    return ridge_mod,clf

def predict_y(model,X_test):
    y_pred = model.predict(X_test)
    return y_pred

def accuracy(y_test,y_pred):
    acc_score = accuracy_score(y_test,y_pred)
    return acc_score

def main():
    X,y= load_data('breast_cancer.csv')
    X_train, X_test, y_train, y_test = splitting_data(X,y)
    X_train, X_test = scaling_data(X_train,X_test)
    ridge_mod,clf = train_model(X_train,y_train)
    y_pred = predict_y(ridge_mod,X_test)
    acc_score = accuracy(y_test,y_pred)
    print("Accuracy score is: ",acc_score)
    print("Ridge Thetas:", clf.coef_)


if __name__=='__main__':
    main()

