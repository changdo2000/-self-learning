from numpy.core.numeric import ones
from numpy.lib.npyio import genfromtxt, load
from numpy.ma import concatenate, where
import pandas as pd
import numpy as np

def load_data(file_name):
    data = genfromtxt(file_name, delimiter = ',')
    X = data[:,0:-1]
    y = data[:,-1]
    return X, y

def predict(X, w):
    return 1.0 / (1.0 + np.exp(-np.dot(X, w)))

def logistic_regression(X, y, epochs, learning_rate):
    Xbar = np.concatenate((X, np.ones((len(X), 1))), axis = 1)
    w = np.zeros(Xbar[0]. shape).T
    epoch = 0
    while epoch < epochs:
        print("Epoch: ", epoch)
        epoch += 1
        shuffle_data = np.random.permutation(len(X))
        for id_data in shuffle_data:
            yhat = predict(Xbar[id_data], w)
            w = w + (learning_rate * (y[id_data] - yhat) *  Xbar[id_data]).T
        
    return w

def accuracy(X, y, w):
    Xbar = np.concatenate((X, np.ones((len(X), 1))), axis = 1)
    cnt = 0
    for i in range(len(X)):
        if (round(predict(Xbar[i], w)) == y[i]):
            cnt += 1
    return cnt / len(X)

if __name__ == '__main__':
    file_name = 'data.csv'
    X, y = load_data(file_name)
    w = logistic_regression(X, y, 10000, 0.005)
    print(w)
    print(accuracy(X, y, w))
    print(len(X))
    