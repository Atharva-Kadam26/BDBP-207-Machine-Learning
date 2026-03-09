import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


data = pd.read_csv("breast_cancer.csv")

X = data.iloc[:,:-1]
y = data.iloc[:,-1]

oe = OneHotEncoder(sparse_output=False)
X_encoded = oe.fit_transform(X)

le= LabelEncoder()
y_encoded = le.fit_transform(y)

X_train,X_test,y_train,y_test = train_test_split(X_encoded,y_encoded,test_size=0.3,random_state=999)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
m = model.fit(X_train,y_train)
y_pred = model.predict(X_test)

print("Predicted y values are:" ,y_pred)
print("Actual y values are:" ,y_test)
print("Accuracy score: ", accuracy_score(y_test,y_pred)*100)
