"""Implement entropy measure using Python. The function should accept a set of data
points and their class labels and return the entropy value."""

import numpy as np
import pandas as pd
from collections import Counter
import math

#Load dataset
data = pd.read_csv('breast-cancer_2.csv')

#Column names
columns=["age","menopause","tumor_size","inv_nodes","node_caps","deg_malig","breast","breast_quad","irradiation","class"]

data.columns = columns

#removing quotes from the dataset
for i in data.columns:
    data[i] = data[i].str.replace("'","")  #replacing ' with nothing

print(data)


"""Compute entropy of dataset"""

labels = data["class"].tolist()  #selecting the target column

def entropy(labels):
    tot_samples = len(labels)
    count_of_labels = Counter(labels)

    entropy=0
    for count in count_of_labels.values():
        probability = count / tot_samples
        entropy -= probability * math.log2(probability)

    return entropy

parent = data["class"].tolist()

children=[]

for value in data["age"].unique():
    subset = data[data["age"] == value]["class"].tolist()
    children.append(subset)

#computing weighted entropy...
parent_entropy = entropy(parent)

weighted_entropy = 0
total_no_of_samples=len(parent)

for child in children:
    weighted_entropy += (len(child)/total_no_of_samples) * entropy(child)

info_gain = parent_entropy - weighted_entropy

print("information_gain (age) :", info_gain)



