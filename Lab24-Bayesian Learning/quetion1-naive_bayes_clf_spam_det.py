"""2.​ Implement Naive Bayes classifier for spam detection using scikit-learn library - use the
dataset from
https://www.kaggle.com/datasets/vishakhdapat/sms-spam-detection-dataset/data"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def load_data(filepath):
    df = pd.read_csv(filepath,encoding ='latin-1')
    # print(df.columns.tolist())
    # print(df.head(3))
    df.columns = ['label','message']
    return df

def exploring_data(df):
    print(f"Total messagees: {len(df)}")
    print("Spam Messages:",(df['label']=='spam').sum())
    print("Ham Messages:",(df['label']=='ham').sum())
    print(f"\n Sample rows:")
    print(df.head(3))


from sklearn.model_selection import train_test_split
def pre_processing(df):
    df['label'] = df['label'].map({'spam':1,'ham':0})
    print(df.head(3))
    return df

#Since the dataset contains data in the form of texts >>>
#we use vectorizer to convert text into numeric data

from sklearn.model_selection import train_test_split

def splitting(df):
    X_train,X_test,y_train,y_test = train_test_split(df['message'],df['label'],test_size=0.2)
    return X_train,X_test,y_train,y_test

def vectorizing(X_train,X_test):
    vectorize = TfidfVectorizer(
        stop_words='english',     # ignoring words like the,is,a
        max_features=5000         # when we vectorize the text the model looks at unique words in all the messages.
        # but if there are many unique words then we only take the top words accordinhg to their TF-IDF score in our case we have taken 5000 as our number of words to be taken
    )
    X_train_vec = vectorize.fit_transform(X_train)
    X_test_vec = vectorize.transform(X_test)
    return X_train_vec,X_test_vec

from sklearn.naive_bayes import MultinomialNB

def naive_bayes(X_train_vec,y_train):
    model = MultinomialNB()  #why multinomialNB? i took this bcoz the features are word frequencies - counts of words
    #in a message. this handles the count data the best.the model internally counts how often each word appears in spam vs ham messages and builds its probability tables

    #The model reads through thousands of labelled spam and ham messages and builds two notebooks:
    # 'Words I see in spam' and 'Words I see in ham'.
    # Later it uses these notebooks to judge new messages.

    trained_model =model.fit(X_train_vec,y_train)
    return trained_model


def predict(model,X_test_vec):
    y_pred = model.predict(X_test_vec)  #the one with higher probability is chosen
    return y_pred

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

def evaluation(y_test,y_pred):
    accuracy = accuracy_score(y_test,y_pred) #in this case accuracy alone is not enough
    precision = precision_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)
    return accuracy,precision,recall,f1

from sklearn.metrics import confusion_matrix

def conf_matrix(y_test,y_pred):
    matrix = confusion_matrix(y_test,y_pred)
    print("Confusion Matrix:")

    print(f'True ham (correct): {matrix[0][0]}')
    print(f'False spam (ham but predicted spam): {matrix[0][1]}')
    print(f'False ham (spam but predicted ham): {matrix[1][0]}')
    print(f'True spam (correct): {matrix[1][1]}')


def main():
    df = load_data("spam_sms.csv")
    exploring_data(df)
    print("\n After encoding the labels.:\n")
    df = pre_processing(df)
    X_train, X_test, y_train, y_test = splitting(df)
    X_train_vec,X_test_vec= vectorizing(X_train,X_test)
    trained_model = naive_bayes(X_train_vec,y_train)
    y_pred = predict(trained_model,X_test_vec)
    accuracy,precision,recall,f1 = evaluation(y_test,y_pred)
    print(f"\nAccuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 score: {f1}")

    conf_matrix(y_test,y_pred)


if __name__ == "__main__":
    main()

