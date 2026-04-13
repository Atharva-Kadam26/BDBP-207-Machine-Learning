"""Implement XGBoost regressor using scikit-learn"""
from sklearn.datasets import load_diabetes
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data():
    diabetes = load_diabetes()
    X = diabetes['data']
    y = diabetes['target']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=999)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def model(X_train, y_train):
    xgb_regressor = XGBRegressor(
        n_estimators=60,
        learning_rate=0.1,
        max_depth=1,
        random_state=42
    )
    trained_model = xgb_regressor.fit(X_train, y_train)
    return trained_model

def prediction(trained_model,X_test):
    y_pred = trained_model.predict(X_test)
    return y_pred

def evaluate_model(y_test,y_pred):
    r2 = r2_score(y_test,y_pred)
    return r2

def main():
    X_train, X_test, y_train, y_test = load_data()
    trained_model = model(X_train, y_train)
    y_pred = prediction(trained_model,X_test)
    r2 = evaluate_model(y_test,y_pred)
    print("r2 score:" , r2)


if __name__ == '__main__':
    main()
