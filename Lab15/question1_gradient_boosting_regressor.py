"""Implement Gradient Boost Regression and Classification using scikit-learn. Use the
Boston housing dataset from the ISLP package for the regression problem and weekly
dataset from the ISLP package and use Direction as the target variable for the
classification."""
from ISLP import load_data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score


def data(filename):
    data = load_data(filename)
    X = data.drop("medv", axis=1)
    y = data["medv"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=999)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def model(X_train, y_train):
    model=GradientBoostingRegressor(n_estimators=100,learning_rate=0.1,max_depth=4,random_state=999)
    trained_model=model.fit(X_train,y_train)
    return trained_model

def predicting_values(X_test, trained_model):
    y_pred = trained_model.predict(X_test)
    return y_pred

def evaluating_model(y_test, y_pred):
    r2 = r2_score(y_test, y_pred)
    return r2

def main():
    X_train, X_test, y_train, y_test = data("Boston")
    trained_model = model(X_train, y_train)
    y_pred = predicting_values(X_test, trained_model)
    r2 = evaluating_model(y_test, y_pred)

    print("The predicted y values are:", y_pred)
    print("The original y values are:", y_test.tolist())
    print("R2 score is :", r2)

if __name__ == "__main__":
    main()











