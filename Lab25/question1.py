"""3.​ Use CIFAR10 dataset and develop a ML model for image classification using kNN"""
from sklearn.neighbors import KNeighborsClassifier
#use logistic regression instead of KNN

from tensorflow.keras.datasets import cifar10
import numpy as np

def load_data():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()  # y did we do like this?? that's bcoz the cifar dataset comes pre split already

    print(f"training images shape: {X_train.shape}")
    print(f"testing images shape: {X_test.shape}")
    print(f"testing labels shape: {y_test.shape}")
    print(f"pixel values range:{X_train.min()} to {X_train.max()}")
    return X_train, y_train, X_test, y_test

import matplotlib.pyplot as plt
import numpy as np

Class_names  = ['airplane', 'automobile', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# def view_images(X_train, y_train, num_images=25):
#     # pick 25 random images
#     indices = np.random.randint(0, len(X_train), num_images)
#
#     plt.figure(figsize=(10, 10))
#
#     for i, idx in enumerate(indices):
#         plt.subplot(5, 5, i + 1)  # 5x5 grid
#         plt.imshow(X_train[idx])  # show the image
#         plt.title(CLASS_NAMES[y_train[idx][0]], fontsize=9)
#         plt.axis('off')  # hide axis numbers
#
#     plt.suptitle('Sample CIFAR10 Images', fontsize=14)
#     plt.tight_layout()
#     plt.show()


def preprocessing(X_train,y_train, X_test,y_test):
    #we flatten the vectors of the images

    #flattening is necessary bcoz knn requires 1D vector to compute euclidean distance
    X_train_f = X_train.reshape(X_train.shape[0], -1)
    X_test_f = X_test.reshape(X_test.shape[0], -1)

    #normalising
    #we are normalising bcoz the pixels with value 255 shld not dominate the distance calculation compared to pixels with val 10

    X_train_n = X_train_f.astype('float32')/255.0
    X_test_n = X_test_f.astype('float32')/255.0

    #flattening the labels
    y_train_f = y_train.flatten()
    y_test_f = y_test.flatten()

    return X_train_n, X_test_n, y_train_f, y_test_f

def subset(X_train, y_train,n_samples = 10000):
    """
    we are taking the subset of the training data because the dataset contains 50000 samples and 3072 features which will
    be very slow thus we use random subset of 10000 samples"""

    indices = np.random.choice(
        len(X_train),n_samples,replace=False
    )
    return X_train[indices], y_train[indices]


def training_model(X_train, y_train,k):
    model = KNeighborsClassifier(
        n_neighbors=k, #no. of neighbours to vote
        metric='euclidean',  #how to measure distance
        n_jobs=-1  #using all cpu cores
    )
    trained_model =model.fit(X_train, y_train)

    return trained_model

#Hyperparameter tuning

from sklearn.metrics import accuracy_score

def hyperparam_tuning(X_train_n, y_train_f,X_test_n,y_test_f):
    best_k =1
    best_acc = 0

    for k in [1,3,5,7,9,11]: #why are the values odd??
        #bcoz odd numbers avoid ties in voting
        model = KNeighborsClassifier(
            n_neighbors=k, metric='euclidean', n_jobs=-1
        )
        t_model = model.fit(X_train_n, y_train_f)
        y_pred = t_model.predict(X_test_n)
        accuracy = accuracy_score(y_test_f, y_pred)
        print(f'k={k} Accuracy: {accuracy*100:.2f}%')

        if accuracy > best_acc:
           best_k = k
           best_acc = accuracy

    print(f'\n Best k = {best_k} Accuracy: {best_acc*100:.2f}%')
    return best_k

from sklearn.metrics import classification_report,confusion_matrix
def evaluate(trained_model,X_test,y_test):
    y_pred = trained_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n Overall Accuracy: {acc*100:.2f}%" )

    print(classification_report(y_test, y_pred,target_names=Class_names))

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    return y_pred

def main():
    X_train,y_train,X_test,y_test = load_data()
    # view_images(X_train, y_train)
    X_train_n,X_test_n,y_train_f,y_test_f = preprocessing(X_train, y_train, X_test, y_test)
    X_train_subset, y_train_subset = subset(X_train_n,y_train_f,n_samples=10000)
    # print({f"{len(X_train_subset)} training samples"})

    trained_model = training_model(X_train_subset,y_train_subset,k=5)

    best_k = hyperparam_tuning(X_train_n,y_train_f, X_test_n, y_test_f)

    y_pred = evaluate(trained_model,X_test_n,y_test_f)






if __name__ == '__main__':
    main()