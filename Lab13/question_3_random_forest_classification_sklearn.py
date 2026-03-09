"""Implement Random Forest algorithm for classification using scikit-learn. Use iris dataset."""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import  RandomForestClassifier
from sklearn.metrics import accuracy_score

def load_data():
    diabetes = load_iris()
    X = diabetes.data
    y = diabetes.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=999)
    return X_train, X_test, y_train, y_test

def train_random_forest(X_train,y_train):
    model = RandomForestClassifier(n_estimators=100 ,max_depth = None, random_state=999)
    model.fit(X_train, y_train)
    return model


def predict(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred

def evaluate_model(model, X_test, y_test):
    y_pred = predict(model, X_test)
    acc_score = accuracy_score(y_test, y_pred)
    return y_pred, acc_score

def main():
    X_train, X_test, y_train, y_test = load_data()
    model = train_random_forest(X_train, y_train)
    y_pred = predict(model, X_test)
    acc_score = accuracy_score(y_test, y_pred)
    print("Ground truth y values are :", y_test)
    print("\nPredicted y values are :", y_pred)
    print("\nAccuracy score:", acc_score)

if __name__ == '__main__':
    main()
