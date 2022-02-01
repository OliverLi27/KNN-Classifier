'''
Created on Jan 25, 2022
@author: Xingchen Li
'''


import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Normalize the data set X
def normalize(X, axis=-1, p=2):
    lp_norm = np.atleast_1d(np.linalg.norm(X, p, axis))
    lp_norm[lp_norm == 0] = 1
    return X / np.expand_dims(lp_norm, axis)

# Standardized data set X
def standardize(X):
    X_std = np.zeros(X.shape)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    # The denominator can't equal 0
    # X_std = (X - X.mean(axis=0)) / X.std(axis=0)
    for col in range(np.shape(X)[1]):
        if std[col]:
            X_std[:, col] = (X[:, col] - mean[col]) / std[col]
    return X_std

def shuffle_data(X, y, seed):
    if seed:
        np.random.seed(seed)
    idx = np.arange(len(y))
    np.random.shuffle(idx)    
    return X[idx], y[idx]

def train_test_split(X, y, test_size = 0.2, shuffle = True, seed = None):
    if shuffle:
        X, y = shuffle_data(X, y, seed)
    n_train_samples = int(len(y) * (1 - test_size))
    X_train, y_train = X[:n_train_samples], y[:n_train_samples]
    X_test, y_test = X[n_train_samples:], y[n_train_samples:]
    return X_train, y_train, X_test, y_test 

def acc(y, y_pred):
    return np.sum(y == y_pred) / len(y)

class KNN(object):
    def __init__(self, k):
        self.k = k    
    # Calculate Euclidean distances between test samples and all training samples
    def dist(self, sample, dataset):
        return np.sum((dataset - sample) ** 2, axis = 1)
    # Sort the distances, and then get the labels corresponding to the first k minimum distances
    def get_knn_labels(self, distances, labels): 
        knn_labels = []
        knn_dist = np.sort(distances)[:self.k]
        for dist in knn_dist:
            label = labels[dist==distances]
            knn_labels.append(label[0])
        return np.array(knn_labels)
    # Vote on k labels, and the first one shall be the same number of votes
    def vote(self, knn_labels):
        knn_labels = knn_labels.tolist()
        find_label, find_count = 0, 0
        for label, count in Counter(knn_labels).items():
            if count > find_count:
                find_count = count
                find_label = label
        return find_label
    # Dist, GEt_KNn_labels and VOTE functions are used for prediction
    def predict(self, X_test, X_train, y_train):
        y_test = []
        for sample in X_test:
            distances = self.dist(sample, X_train)
            knn_labels = self.get_knn_labels(distances, y_train)
            label = self.vote(knn_labels)
            y_test.append(label)
        return np.array(y_test)

if __name__ == "__main__":
    X, y = load_iris(return_X_y=True)
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2)
    X_train = normalize(X_train)
    X_test = normalize(X_test)
    # X_train = standardize(X_train)
    # X_test = standardize(X_test)
    results = []
    for k in range(1, len(X_train) + 1):
        clf = KNN(k)
        y_pred = clf.predict(X_test, X_train, y_train)
        results.append(acc(y_test, y_pred))
    
    plt.xlabel("k")
    plt.ylabel("acc")
    plt.plot([k for k in range(1, len(X_train) + 1)], results)
    # plt.show()  
    plt.savefig("k_acc.png")