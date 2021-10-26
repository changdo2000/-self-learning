from mlxtend.data import loadlocal_mnist
import os
import numpy as np
import matplotlib as mpl
from PIL import Image

def show(X, y):
    print(y)
    img = Image.fromarray(X, 'L')
    img.show()

def softmax(Z):
    x = np.exp(Z - np.ndarray.max(Z))
    return x / np.sum(x)

def softmax1(z):
    exp = np.exp(z - np.max(z))
    for i in range(len(z)):
        exp[i] /= np.sum(exp[i])
    return exp

def stable_softmax(x):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    ssoftmax = numerator/denominator

    return ssoftmax

def one_hot(y, c):
    y_hot = np.zeros((y.size, c))
    y_hot[np.arange(y.size), y] = 1
    return y_hot

def predict(X, w, b):
    z = np.dot(X, w) + b
    y_hat = softmax(z)
    return np.argmax(y_hat)

def train(X, y, lr, eps):

    m, n = X.shape
    c = 10

    w = np.random.random((n, c))
    b = np.random.random(c)

    for epoch in range(eps):
        y_predict = softmax(np.dot(X, w) + b)   
        y_truth   = one_hot(y, c)

        w -= lr * (1.0 / m) * np.dot(X.T, (y_predict - y_truth))
        b -= lr * (1.0 / m) * np.sum(y_predict - y_truth)
        if (epoch % 100 == 0):
            print("eps: ", epoch)
    return w, b


if __name__ == '__main__':

    #read data
    imgs_train_path = str('C:/Users/Admin/Desktop/-self-learning/ml-basic/classification-writing-number/')
    imgs_test_path = str('C:/Users/Admin/Desktop/-self-learning/ml-basic/classification-writing-number/')

    X_train, y_train = loadlocal_mnist(images_path = imgs_train_path +  'train-images.idx3-ubyte', labels_path = imgs_train_path + 'train-labels.idx1-ubyte')
    X_test, y_test  = loadlocal_mnist(images_path = imgs_test_path + 't10k-images.idx3-ubyte', labels_path = imgs_test_path + 't10k-labels.idx1-ubyte')
    #   show(X_train[0].reshape((28, 28)), y_train[0])
    print("Loading...!")

    W, b = train(X_train, y_train, 0.9, 1000)
    cnt = 0
    for i in range(0, 10000):
        if (predict(X_test[i], W, b) == y_test[i]):
            cnt += 1
    print("Acc: ", cnt / 10000)
