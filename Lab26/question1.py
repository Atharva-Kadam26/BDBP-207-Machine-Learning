"""Learn XAI - SHAP. Use SHAP in one of your models and plot beeswam plot. Study the
influencing features. - https://shap.readthedocs.io/en/latest/"""

import shap
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = load_breast_cancer()

X = pd.DataFrame(data.data,columns= data.feature_names)
y = data.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)

print("Accuracy:",accuracy)

explainer = shap.Explainer(model.predict,X_train)
shap_values = explainer(X_test)

shap.plots.beeswarm(shap_values)

plt.show()