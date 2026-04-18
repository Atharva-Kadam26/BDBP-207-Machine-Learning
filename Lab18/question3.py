"""Implement twitter sentiment prediction using SVM - Try different kernel functions and
compare the results.
https://www.kaggle.com/code/langkilde/linear-svm-classification-of-sentiment-in-tweets/n
otebook"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def load_data():
    df = pd.read_csv('Tweets.csv')
    return df

def preprocessing(df):
    df = df[["text","airline_sentiment"]] # only needed columns
    df = df.dropna()
    return df


def encoding(df):
    df["airline_sentiment"] = df["airline_sentiment"].map({
        'negative':0,
        'neutral':1,
        'positive':2
    })
    return df

def vectorize(X_train, X_test):
    vectorizer = TfidfVectorizer(stop_words='english',max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return X_train_vec, X_test_vec

def train_svm(X_train,y_train,kernel):
    model=SVC(kernel=kernel)
    model.fit(X_train,y_train)
    return model

def evaluation(model,X_test,y_test):
    y_pred=model.predict(X_test)
    acc= accuracy_score(y_test,y_pred)
    print("Accuracy:",acc)
    print(classification_report(y_test,y_pred))
    return acc

def compare_kernels(X_train, X_test, y_train, y_test):
    kernels = ['linear','rbf','poly']
    results={}

    for k in kernels:
        print(f"\n Kernel: {k}")
        model = train_svm(X_train,y_train,k)
        acc = evaluation(model,X_test,y_test)
        results[k] = acc

    return results

def main():
    df = load_data()
    df = preprocessing(df)
    df = encoding(df)

    X = df['text']
    y = df['airline_sentiment']

    #splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=999)

    X_train_vec, X_test_vec = vectorize(X_train,X_test)

    #comparing kernels
    results = compare_kernels(X_train_vec,X_test_vec,y_train,y_test)
    print("\n Final Comparison:",results)


if __name__ == "__main__":
    main()
