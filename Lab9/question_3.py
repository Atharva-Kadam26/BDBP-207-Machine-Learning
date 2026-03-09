"""Implement a classification decision tree algorithm using scikit-learn for the sonar  dataset."""
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def load_data(filename):
    df = pd.read_csv(filename)
    X= df.iloc[:,:-1]
    y= df.iloc[:,-1]
    feature_names = list(X.columns)
    return X, y ,feature_names

def splitting_data(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=999)
    return X_train, X_test, y_train, y_test

def model(X_train,y_train):
    model = DecisionTreeClassifier(random_state=42)
    mod = model.fit(X_train, y_train)
    return mod

def tree_structure(mod,feature_names):
    """Creating the tree structure"""
    nodes = mod.tree_.node_count
    left_child = mod.tree_.children_left
    right_child = mod.tree_.children_right
    feature = mod.tree_.feature   #feature used for splitting
    threshold = mod.tree_.threshold   #split value
    value = mod.tree_.value    #this is for the values in the nodes

    node_depth = np.zeros(shape=nodes,dtype=np.int64)   #depth of each node

    is_leaf = np.zeros(shape=nodes,dtype=np.bool)     #this will mark whether a node is a leaf node or an internal split node

    stack = [(0,0)]   #[(node_id,depth)]

    """Traversing through the tree"""
    while len(stack) > 0:
        node_id,depth = stack.pop()
        node_depth[node_id] = depth   #records how deep the node is in the tree
        is_split_node = left_child[node_id] != right_child[node_id]

        if is_split_node:
            stack.append((left_child[node_id],depth+1))
            stack.append((right_child[node_id],depth+1))

        else:
            is_leaf[node_id] = True

        

def prediction(X_test,mod):
    y_pred = mod.predict(X_test)
    return y_pred

def evaluation(y_test,y_pred):
    accuracy = accuracy_score(y_test,y_pred)
    return accuracy

def main():
    X, y ,feature_names = load_data("sonar data.csv")
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    mod = model(X_train, y_train)
    y_pred = prediction(X_test, mod)
    accuracy = evaluation(y_test,y_pred)
    print("Predicted values:", y_pred)
    print("Actual values:", y_train.values)
    print("Accuracy Score:",accuracy)


if __name__ == '__main__':
    main()