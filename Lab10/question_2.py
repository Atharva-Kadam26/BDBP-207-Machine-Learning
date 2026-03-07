"""Implement information gain measures. The function should accept data points for
parents, data points for both children and return an information gain value."""

import numpy as np
import pandas as pd
import math
from collections import Counter

data = pd.read_csv("breast-cancer_2.csv")

columns=["age","menopause","tumor_size","inv_nodes","node_caps","deg_malig","breast","breast_quad","irradiation","class"]

data.columns = columns

#removing quotes from the dataset
for i in data.columns:
    data[i] = data[i].str.replace("'","")  #replacing ' with nothing

print(data)

X = data.iloc[:,:-1]
y = data.iloc[:,-1]

print("Total no. of sample:" ,len(y))

def entropy(labels):
    counts = Counter(labels)
    total_samples = len(labels)

    entropy_val = 0

    for count in counts.values():

        prob = count/total_samples
        entropy_val -= prob * math.log2(prob)

    return entropy_val

dataset_entropy = entropy(y)
print("Entropy of the dataset:",dataset_entropy)

parent_labels = data['class']
left_labels = data[data['node_caps']=='yes']['class']
right_labels = data[data['node_caps']=='no']['class']

def information_gain(parent_labels,left_labels,right_labels):
    parent_entropy = entropy(parent_labels)

    weight_left = len(left_labels) / len(parent_labels)
    weight_right = len(right_labels) / len(parent_labels)

    left_entropy = entropy(left_labels)
    right_entropy = entropy(right_labels)

    information_gain = parent_entropy - ( weight_left * left_entropy + weight_right * right_entropy )

    return information_gain

ig = information_gain(parent_labels,left_labels,right_labels)
print(len(left_labels))
print(len(right_labels))
print("Information Gain:", ig)

