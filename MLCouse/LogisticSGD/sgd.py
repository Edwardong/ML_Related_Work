#Zheren Dong zherendong@ucsb.edu

from math import exp
import random
from data import load_adult_train_data, load_adult_valid_data

# TODO: Calculate logistic
def logistic(x):
    return 1 / (1 + exp(-x))

# TODO: Calculate dot product of two lists
def dot(x, y):
    s = 0
    for k in range(len(x)):
        s += x[k] * y[k]
    return s

# TODO: Calculate prediction based on model
def predict(model, point):
    d = dot(model, point["features"])
    return logistic(d)

# TODO: Calculate accuracy of predictions on data
def accuracy(data, predictions):
    correct = 0
    for i in range(len(predictions)):
        label = 1
        if predictions[i] <= 0.5:
            label = 0
        if data[i]["label"] == label:
            correct += 1
    return float(correct)/len(data)

# TODO: Update model using learning rate and L2 regularization
def update(model, point, rate, lam):
    y_hat = predict(model, point)
    for i in range(len(model)):
        model[i] = model[i] + rate *(-lam * model[i] + point["features"][i] * (point["label"] - y_hat))
    return model

def initialize_model(k):
    return [random.gauss(0, 1) for x in range(k)]

# TODO: Train model using training data
def train(data, epochs, rate, lam):
    model = initialize_model(len(data[0]['features']))
    for epoch in range(epochs):
        # TODO: make learning rate decreasing over time. decay is a hyper-parameter to control decreasing rate with time in range[0.0, 1.0]
        decay = 0
        lr = rate * 1.0 / (1.0 + decay * epoch)
        for i in range(len(data)):
            model = update(model, data[i], lr, lam)
    return model
        
def extract_features(raw):
    data = []
    for r in raw:
        point = {}
        point["label"] = (r['income'] == '>50K')

        features = []
        features.append(1.)
        features.append(float(r['age'])/100)
        features.append(float(r['education_num'])/20)
        features.append(r['marital'] == 'Married-civ-spouse')
        #TODO: Add more feature extraction rules here!
        features.append(r['capital_gain'] != '0')
        features.append(float(r['hr_per_week']) / 100)
        features.append(r['race'] == 'White')
        features.append(r['education'] == 'Masters' or r['education'] == 'Doctorate')
        point['features'] = features
        data.append(point)
    return data

# TODO: Tune your parameters for final submission
def submission(data):
    # TODO: a simple version of grid-search
    # valid_data = extract_features(load_adult_valid_data())
    # model = initialize_model(len(data[0]['features']))
    # predictions = [predict(model, p) for p in valid_data]
    # acc = accuracy(valid_data, predictions)
    # best_epoch = 0
    # best_rate = 0
    # best_lam = 0
    # for epoch in [10, 20, 50, 100]:
    #     for rate in [0.001, 0.01, 0.1]:
    #         for lam in [0, 0.001, 0.1, 1]:
    #             tmp = train(data, epoch, rate, lam)
    #             predictions = [predict(tmp, p) for p in valid_data]
    #             if accuracy(valid_data, predictions) > acc:
    #                 acc = accuracy(valid_data,predictions)
    #                 model = tmp

    #                 best_epoch = epoch
    #                 best_rate = rate
    #                 best_lam = lam
    # print best_epoch, best_rate, best_lam               
    # return model


    #After runing the code above, I found that epoch = 20, rate = 0.01 and lam =0 can give the highest accuracy.
    return train(data,20,0.01,0)
    
