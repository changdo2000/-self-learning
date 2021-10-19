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
    return np.exp(Z) / sum(np.exp(Z))

def ReLU(x):
    if x > 0:
        return x
    else:
        return 0

def predict():
    return

def train(X, y, lr):
    return


if __name__ == '__main__':

    #read data
    imgs_train_path = str('C:/Users/Admin/Desktop/-self-learning/ml-basic/classification-writing-number/')
    #X_train, y_train = loadlocal_mnist(images_path = imgs_train_path +  'train-images.idx3-ubyte', labels_path = imgs_train_path + 'train-labels.idx1-ubyte')
    #show(X_train[0].reshape((28, 28)), y_train[0])
    print(ReLU(4))
    a = np.array([1, 2, 3, 4])
    print(softmax(a))