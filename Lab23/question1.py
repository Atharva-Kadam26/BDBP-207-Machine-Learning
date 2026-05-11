"""1.​ Develop prediction model for Iris.csv using joint probability distribution approach
a.​ Use only the first two features, SepalLengthCm, SepalWidthCm and the target
variable
b.​ Add random noise to the features
c.​ Discretize the feature values
d.​ Build a decision tree model with max_depth = 2, then, compare the accuracy of
this model with the joint probability distribution method"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def load_data():
    iris = load_iris()
    X= iris.data[:,:2]
    y = iris.target
    return X,y

def adding_noise(X,noise_level = 0.3):
    #We add noise to test the robustness of the model whether the model can classify correctly even if the input measurements are slightly off
    noise = np.random.normal(0,noise_level,size=X.shape) #noise is centered at 0 and noise_level=0.5 -> it is the standard deviation makes it easy helps in creating the exact same as X
    return X+noise   #we get a noisier version of X

def discretization(X,bins):
    X_discrete = np.zeros_like(X, dtype= int)
    for col in range(X.shape[1]):
        X_discrete[:,col] = pd.cut(
            X[:,col],bins=bins,labels=False
        )
    return X_discrete

#We are discretizing because the joint prob model works by counting how often each combination of feature values appeared alongside each class.
#But if features are decimals then the model rarely sees the same value twice which makes each value unique

# with the help of discretization we group nearby values into bins or buckets
#eg temperature values we bin them as cold/warm/hot etc it is easier to find patterns in this way


def train_joint_probability_model(X,y):
    classes = np.unique(y)
    joint_prob={}
    for c in classes:
        X_c = X[y==c]  # filters the data to keep only rows belonging to class c.Eg when c=0 we get 50 setosa flower rows y==c creates a list of True/False values used to select rows
        class_prob = len(X_c)/len(X) #calculating prob
        feature_probs = {}
        for row in X_c:
            key = tuple(row) # converts rows into tuples as we have to create a dictionary and it cannot be made with lists so converting it to tuples
            feature_probs[key] = (feature_probs.get(key,0)+1) #checks if the feature combination was seen before(if not then defaulting to 0)then adds 1. this counts occurrences of each combination

        for key in feature_probs:
            feature_probs[key] = feature_probs[key]/len(X_c) # converting frequencies to prob divides each count by the total samples in tht class

        joint_prob[c] = {
            'class_prob':class_prob,
            'feature_probs':feature_probs
        }  # storing class prob and feature prob in the dictionary

    # the generative model learns the joint distribution P(X,y) - the prob of features and class together.
    #this is different from the discriminative model like decision tree which learns P(y|X) - the class given features directly

    return joint_prob

def predict_joint_probability(joint_prob,X):
    # we use the prob tables learned during training to classify new samples.
    # for each new flower it asks: given these feature values, which class makes this combination most likely?
    # It picks the class with the highest joint probability

    predictions=[] #empty list tht will collect one prediction per sample. At the end this becomes our array of predicted labels
    for row in X: #loops thru every test sample one at each time each row contains the  bin values for that sample's two features
        key = tuple(row)
        best_class = None #initialising tracking variables
        best_prob = -1
        for c, data in joint_prob.items(): # loops thru all 3 classes. For each class we compute the joint probability of the sample belonging to it

            feature_prob = data['feature_probs'].get(
                key,1e-6  #looks up P(features|class=c) from the training set. If the combination was never seen instead of 0 we give 1e-6
            )
            prob = feature_prob * data['class_prob'] # computing the joint probability P(features|class) x P(class).

            if prob > best_prob: # if the current class gives the highest probability than any other class checked out so far, we update our best guess
                best_class = c
                best_prob = prob
        predictions.append(best_class)

    return np.array(predictions) # converts python list of predictions into a NumPy array - needed for accuracy score to work correctly

# -----------------------------------------------------------
#Comparing the generative model with decision tree discriminative model
# -----------------------------------------------------------


def train_decision_tree_model(X_train,y_train):
    dt = DecisionTreeClassifier( max_depth=2,random_state=42)
    dt.fit(X_train,y_train)
    return dt

def compare_models(X,y):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    joint_prob = train_joint_probability_model(X_train,y_train)
    y_pred_joint = predict_joint_probability(
        joint_prob ,X_test
    )

    acc_joint = accuracy_score(y_test,y_pred_joint)
    dt_model = train_decision_tree_model(X_train,y_train)
    y_pred_dt = dt_model.predict(X_test)
    acc_dt = accuracy_score(y_test,y_pred_dt)
    print(f"Joint Probability: {acc_joint*100:.2f}%")
    print(f"Decision Tree: {acc_dt*100:.2f}%")
    return acc_joint,   acc_dt


def main():
    X,y = load_data()
    X_noisy = adding_noise(X, noise_level=0.1)
    print("Random noise added")

    # print("\n--- Trying different bin values ---")
    # for bins in [3, 4, 5, 6, 7, 8, 10, 12, 15, 20]:
    #     X_discrete = discretization(X_noisy, bins=bins)
    #     acc_joint, acc_dt = compare_models(X_discrete, y)
    #     print(f"\nbins={bins:2d} → Joint: {acc_joint * 100:.1f}%  DT: {acc_dt * 100:.1f}%")

    X_discrete = discretization(X_noisy, bins=6)
    print("Features discretized into 6 bins")

    print("\n--- Model Comparison ---")
    compare_models(X_discrete, y)


if __name__ == "__main__":
    main()

#the joint prob model needs to have the exact same(SepalLength bin, SepalWidth bin) combo during training to give a good
# prob. If a test sample has a new combination, it falls back to the tiny 1e-6 value i.e a weak guess

# Noise + discretization into only 5 bins means some information is lost. The joint probability model suffers more from this because it relies on exact bin combinations. The decision tree is less sensitive to this.

# The decision tree learned a threshold rule like "if SepalLength bin > 2 → likely Virginica." This rule applies to all the values
#— not just ones seen in training. So it generalises better to new samples.