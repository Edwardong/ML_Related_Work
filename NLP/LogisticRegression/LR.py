import numpy as np
import math
from math import log
import random
import time

def run(Xtrain_file, Ytrain_file, test_data_file, pred_file):
    train(Xtrain_file, Ytrain_file, test_data_file, pred_file, True)

#helper functions below
def sigmoid(x):
    return 1/(1+math.e**(-x))

def read_data(x_path, y_path):
    X = []
    with open(x_path, "r+") as in_f:
        for line in in_f:
            X.append(np.array([int(x) for x in line.split(",")]))
    Y = []
    with open(y_path, "r+") as in_f:
        for line in in_f:
            Y.append(int(line.replace("\n","")))
    return X,Y

def write_data(test_x_path, test_y_path,w):
    test_X = []
    with open(test_x_path, "r") as in_f:
        for line in in_f:
            test_X.append(np.array([int(x) for x in line.split(",")]))
        
    result = predict(w, test_X)

    with open(test_y_path, "w+") as out_f:
        for r in result:
            out_f.write(str(r)+"\n")

def train(x_path = "Xtrain.csv", y_path = "Ytrain.csv", test_x_path = None, test_y_path = None, has_test = False):
    X,Y = read_data(x_path,y_path)

    assert len(X) == len(Y)
    feature_dimension = len(X[0])
    train_size = len(X)
    
    print("train size=%d" % (train_size))
    print("dimension of feature=%d" % (feature_dimension))
    
    #learning rate
    lamb = 0.0001
    alpha = 0.01
    decay = 0.95

    training_indices = [ i for i in range(train_size) ]
    # break_threshold = 0.00001
    w = np.random.rand(feature_dimension)
    #w = np.zeros(len(X))
    current_epo = 0
    prev_loss = 0
    progress_threshold = 0.0001
    low_progress_count = 0
    #how many epochs contribute to one decay
    max_low_count = 2
    total_epoch = 2000


    while current_epo < total_epoch:
        current_epo += 1

        #need to shuffle first
        random.shuffle( training_indices )
        
        for i in training_indices:
            x = X[i]
            y = Y[i]
            p = sigmoid(np.dot(w,x))
            #w=w+alphe((y-p)x-2lambda * w)
            w = w + alpha * ( (y-p) * x - 2 * lamb * w )        


        loss = - lamb * np.linalg.norm(w)**2
        
        for i in range(len(X)):
            x = X[i]
            y = Y[i]
            p = sigmoid(np.dot(w,x))
            loss += y * log( p ) + (1-y)* log( 1-p )
                      
        #update alpha if needed
        if loss - prev_loss < 0 :
            low_progress_count += 2
        elif abs(loss - prev_loss) < progress_threshold:
            low_progress_count += 1
        else:
            low_progress_count -= 1

        if low_progress_count >= max_low_count:
            alpha = alpha * decay
            low_progress_count = 0 

        prev_loss = loss

    if not has_test:
        return
    
    write_data(test_x_path,test_y_path,w)
        
def predict(w,X, threshold = 0.5):
    results = []
    for i in range(len(X)):
        x = X[i]
        p = sigmoid(np.dot(w,x))
        if p > threshold:
            res = 1
        else:
            res = 0
        results.append(res)
        
    return results