"""Perform 10-fold cross validation for SONAR dataset in scikit-learn using logistic
regression. SONAR dataset is a binary classification problem with target variables as
Metal or Rock. i.e. signals are from metal or rock."""
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd

def load_data(filename):
    df = pd.read_csv(filename)
    Xi= df.iloc[:,:-1].values
    yi = df.iloc[:,-1].values
    return Xi,yi

def scale_data(X):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X


def logistic_regression(X,y):
    model = LogisticRegression(max_iter=1500)
    accuracies_of_each_fold=[]

    """Doing the cross validation"""
    kf = KFold(n_splits=10, shuffle=True,
               random_state=999)  # Shuffle=True is for randomly shuffling the test and train sets

    for i , j in kf.split(X):
        X_train, X_test =X[i,:],X[j,:]
        y_train, y_test = y[i],y[j]
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        accur = accuracy_score(y_test,y_pred)
        accuracies_of_each_fold.append(accur)

    return accuracies_of_each_fold


def main():
    X,y=load_data("sonar data.csv")
    acc_scores=logistic_regression(X,y)
    print(acc_scores)
    mean_acc=np.mean(acc_scores)
    print("Mean accuracy score is:",mean_acc)


if __name__ == '__main__':
    main()


