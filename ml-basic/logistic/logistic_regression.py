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
        yhat = predict(Xbar, w)
        w = w + (learning_rate * (yhat - y) * Xbar).T
    
    return w


if __name__ == '__main__':
    file_name = 'data.csv'
    X, y = load_data(file_name)

    w = logistic_regression(X, y, 1000, 0.01)
    print(w)
    