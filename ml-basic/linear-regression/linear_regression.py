import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import append
#from activation_function import *

def load_data():
    X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
    y = y = np.array([[ 49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T
    return X, y

def linear_regression(X, y):
    X_train = [np.append(single_data, 1) for single_data in X]
    w = np.zeros((X_train[0].T).shape, dtype = dfloat)

    return 0

if __name__ == "__main__":
    X, y = load_data()
    X_train = [np.append(single_data, 1) for single_data in X]
    w = np.zeros((X_train[0].T).shape, dtype = float)
    print(w)