import numpy as np
import math
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

iris = load_iris()

def load_data(file):
    X=file.data
    y=file.target

    #Convert to Binary Classification
    y = np.where(y==0,-1,1)
    return X,y

def splitting_data(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=999)
    return X_train, X_test, y_train, y_test

def init_sample_weights(X_train):
    num_samples = X_train.shape[0]
    weights = np.ones(num_samples)/num_samples    ##eg x1:1,2,4,5 weight= 1/4 -> 0.25
    return weights

def model_training(X_train, y_train,weights):
    decision_stump = DecisionTreeClassifier(max_depth=1)   # max_depth = 1 because we are using weak learners
    d_s = decision_stump.fit(X_train, y_train,sample_weight = weights)
    return d_s

def predicting_y(X_train,d_s):
    prediction = d_s.predict(X_train)
    return prediction

def weight_error(prediction,y_train,weights):
    misclassified = prediction != y_train    #since we penalize for misclassification
    error = np.sum(weights * misclassified)

    #Preventing divisio by zero
    error = max(min(error,1 - 1e-10),1e-10)
    return error


def model_updating_weights(weights,prediction,error,y_train):

    lam = 0.5 * math.log2((1-error)/error) #lam is higher than error is smaller

    updated_weights = []

    for i in range(len(weights)):

        if prediction[i] == y_train[i]:     #Correct Classification
            new_w = weights[i] * math.exp(-lam)
            print(f"Sample {i} : correct - w * e^(-lam) = {new_w}")

        else:
            new_w = weights[i] * math.exp(lam)   #Misclassified
            print(f"Sample {i} : incorrect - w * e^(lam) = {new_w}")

        updated_weights.append(new_w)

    updated_weights = np.array(updated_weights)

    print("\nweights before normalization:\n",updated_weights)

    #Normalizing weights

    updated_weights = updated_weights / np.sum(updated_weights)
    print("\nweights after normalization:\n",updated_weights)

    return updated_weights,lam

def final_prediction(X_test,models,lam):
    all_preds=[]

    for model in models:
        all_preds.append(model.predict(X_test))

    all_preds = np.array(all_preds)

    weighted_sum = np.dot(lam,all_preds)

    final_pred = np.sign(weighted_sum)
    return final_pred


def main():
    X,y = load_data(iris)
    X_train, X_test, y_train, y_test = splitting_data(X,y)
    weights = init_sample_weights(X_train)

    num_estimators = 10  # number of weak learners

    models = []  # storing trained trees
    lamb = []  # importance of each model is stored as alphas

    for i in range(num_estimators):
        print("Iteration ",i + 1)
        d_s = model_training(X_train, y_train,weights)

        prediction = predicting_y(X_train,d_s)

        error = weight_error(prediction,y_train,weights)
        print("weighted error:",error)

        updated_weights,lam = model_updating_weights(weights,prediction,error,y_train)
        models.append(d_s)
        lamb.append(lam)

    y_pred = final_prediction(X_test,models,lamb)

    acc =accuracy_score(y_test,y_pred)

    print("Accuracy score:" , acc)




if __name__ == "__main__":
    main()

