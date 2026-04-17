"""Implement twitter sentiment prediction using SVM - Try different kernel functions and
compare the results.
https://www.kaggle.com/code/langkilde/linear-svm-classification-of-sentiment-in-tweets/n
otebook"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,classification_report

def load_data():
    df = pd.read_csv("Tweets.csv")
    X = df["text"].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3,random_state = 999)
    return X_train,X_test,y_train,y_test

def 

def main():
    X_train,X_test,y_train,y_test = load_data()


if __name__ == "__main__":
    main()
