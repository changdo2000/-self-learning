from random import expovariate
import numpy as np
import pandas as pd
import math
from csv import reader


 
# Load a CSV file
def load_csv(filename):
	dataset = []
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

def predict(row, coeff):
    yhat = coeff[0]
    for i in range(len(coeff) - 1):  
        yhat += row[i] * coeff[i + 1]
    return 1.0 / (1.0 + np.exp(-yhat))

def log_loss(y, yhat):
    return -(yhat * log(y) + (1 - yhat) * log(1-y))

def logistic_regression(X, y, number_epoch, learning_rate):
    number_coeff = len(X[0]) + 1
    coeff = [0.0 for i in range(number_coeff)]
    epoch = 0
    while epoch < number_epoch:
        epoch += 1
        print("epoch: ", epoch)
        for  i in range(len(X)):
            yhat = predict(X[i], coeff)
            coeff[0] += learning_rate * (y[i] - yhat) * yhat * (1 - yhat)
            for i_coeff in range(1, number_coeff):
                coeff[i_coeff] += learning_rate * (y[i] - yhat) * X[i][i_coeff - 1] * yhat * (1 - yhat)
            
    return coeff

def accuracy(X_train, y_train, coeff):
    cnt = 0
    sum_cnt = 0
    for i in range(len(X_train)):
        sum_cnt += 1
        if (round(predict(X_train[i], coeff)) == y_train[i]):
            cnt += 1
    print(cnt / sum_cnt)

if __name__ == '__main__':
    filename = 'C:/Users/cgb.boo/OneDrive/Desktop/-self-learning/ml-basic/logistic/data.csv'
    dataset  = load_csv(filename)  
    for i in range(len(dataset[0])):
    	str_column_to_float(dataset, i)
    X_train = [row[0:-1] for row in dataset]
    y_train = [row[-1] for row in dataset]
    learning_rate = 0.1
    number_epoch = 1000

    
    coeff = logistic_regression(X_train, y_train, number_epoch, learning_rate)
    print(coeff)
    accuracy(X_train, y_train, coeff)
