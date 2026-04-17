"""For the heart.csv dataset, build a logistic regression classifier to predict the risk of heart disease.
Vary the threshold to generate multiple confusion matrices.  Implement a python code to
calculate the following metrics
● Accuracy
● Precision
● Sensitivity
● Specificity
● F1-score
● Plot the ROC curve
● AUC"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,roc_curve,auc
import matplotlib.pyplot as plt


def load_data():
    data = pd.read_csv('heart.csv')
    X = data.drop('target',axis=1)
    y = data['target']
    return X,y

def split_data(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=999)
    return X_train, X_test, y_train, y_test

def model(X_train,y_train):
    model = LogisticRegression(max_iter=1000)
    trained_model=model.fit(X_train,y_train)
    return trained_model

def predict(trained_model,X_test):
    y_pred = trained_model.predict(X_test)
    y_prob = trained_model.predict_proba(X_test)[:,1]
    return y_pred,y_prob

def calculate_metrics(y_prob,y_test,threshold=[0.3,0.5,0.7]):

    results={}

    for t in threshold:
        y_pred = (y_prob >= t).astype(int)
        #convert probabilities into class labels using threshold
        #eg. prob >= 0.5 -> 1  , prob < 0.5 -> 0

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        #accuracy
        accuracy = (tp+tn)/(tp+tn+fp+fn)

        #precision
        precision  = tp/(tp+fp) if (tp + fp) != 0 else 0

        #sensitivity
        sensitivity = tp/(tp+fn) if (tp + fn) != 0 else 0

        #specificity
        specificity = tn/(tn+fp) if (tn + fp) != 0 else 0

        #f1 score
        f1 = (2 * precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) != 0 else 0

        results[t] = {
        'accuracy':float(accuracy),
        'precision':float(precision),
        'sensitivity':float(sensitivity),
        'specificity':float(specificity),
        'f1 score':float(f1)
        }

    return results

def plot_confusion_matrix(y_test,y_prob,threshold=0.5):

    #Convert probabilities into class labels
    y_pred = (y_prob >= threshold).astype(int)

    #compute confusion matrix
    cm = confusion_matrix(y_test,y_pred)

    plt.figure()
    plt.imshow(cm)
    plt.title(f"Confusion Matrix (Threshold={threshold})")
    plt.colorbar()

    labels = [' 0 (No Disease)','1 (Disease)']

    plt.xticks(np.arange(2),labels)
    plt.yticks(np.arange(2),labels)

    #Show values inside boxes
    for i in range(2):
        for j in range(2):
            plt.text(j,i,cm[i,j],ha='center',va='center')

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def plot_roc(y_test,y_prob):

    #Compute ROC values
    fpr,tpr,thresholds = roc_curve(y_test,y_prob)

    #Compute AUC
    roc_auc = auc(fpr,tpr)

    print("AUC:", round(roc_auc,3))

    #Plot ROC curve
    plt.figure()
    plt.plot(fpr,tpr,label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1],linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

def main():
    X,y = load_data()
    X_train, X_test, y_train, y_test = split_data(X,y)
    trained_model = model(X_train,y_train)
    y_pred,y_prob = predict(trained_model,X_test)
    result = calculate_metrics(y_prob,y_test,threshold=[0.3,0.5,0.7])
    print(result)
    plot_confusion_matrix(y_test,y_pred,threshold=0.5)
    plot_roc(y_test,y_prob)



if __name__ == "__main__":
    main()
