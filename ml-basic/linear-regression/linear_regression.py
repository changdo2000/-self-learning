import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import append
#from activation_function import *

def load_data():
    X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
    y = y = np.array([[ 49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T
    return X, y

def linear_regression(X, y):
    Xbar = np.concatenate((np.ones((X.shape[0], 1)), X), axis = 1)
    return np.dot(np.linalg.pinv(np.dot(Xbar.T, Xbar)), np.dot(Xbar.T, y))

if __name__ == "__main__":
    X, y = load_data()

    w = linear_regression(X, y)
    print(w)

    plt.plot(X.T, y.T, 'ro') 
    plt.xlabel('Height (cm)')
    plt.ylabel('Weight (kg)')
    plt.show()
    
