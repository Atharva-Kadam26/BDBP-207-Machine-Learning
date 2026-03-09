"""Ordinal encoding with sklearn and logistic regression"""
import pandas as pd
import numpy as np

from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv("breast-cancer_2.csv")

columns = [
'age','menopause','tumor_size','inv_nodes','node_caps',
'deg_malig','breast','breast_quad','irradiat','class'
]

data.columns = columns

data = data.replace("?",np.nan)
data = data.dropna()

X=data.iloc[:,:-1]
y=data.iloc[:,-1]

le = LabelEncoder()
y = le.fit_transform(y)

ordinal_encoder = OrdinalEncoder()
X_encoded = ordinal_encoder.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_encoded,y,test_size=0.3,random_state=999)

model = LogisticRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)

print("Accuracy:",accuracy)
