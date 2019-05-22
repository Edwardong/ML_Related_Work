#!/usr/bin/env python

# import the required packages here
import numpy as np
import csv  
import math 

def run(Xtrain_file, Ytrain_file, test_data_file, pred_file):    
    # your implementation here
    # read data from Xtrain_file, Ytrain_file and test_data_file
    # your algorithm
    # save your predictions into the file pred_file


    vs,ws = train(Xtrain_file,Ytrain_file)
    counter = 0 

    #predict on testing data set  
    with open(test_data_file, "r+") as test_f:
        with open(pred_file, "w+") as pred_f:
            for lines in test_f:              
                splited_line = lines.split(",")
                w_vector = np.array([int(x) for x in splited_line])
                prediction = None
                s = 0
                k = 0 
                
                for k in range(len(vs)):
                    s += ws[k] * Vote_p( vs[k], w_vector )                
                if s <= 0 :
                    prediction = 0
                else:
                    prediction = 1
                pred_f.write(str(prediction) + "\n") 
                counter += 1        


def train(Xtrain,Ytrain):
    v_size = None
    cp = None
    vs = None
    ws = None
    k = 0
    T = 20
    for t in range(T):        
        with open(Xtrain, "r+") as x_csvf:
            with open(Ytrain, "r+") as y_csvf:
                counter =0 
                for lines in x_csvf:
                    counter += 1                       
                    # Read data to array                 
                    splited_lines = lines.split(",")
                    w_vector = np.array([int(x) for x in splited_lines]) 
                    #initialize classifier                   
                    if v_size == None:
                        v_size = len(w_vector)
                        vs = [np.zeros(v_size)]
                        ws = [0]
                        cp = vs[0]                     
                    #train using labels    
                    c = int(y_csvf.readline())
                    if c == 0:
                        c = -1
                    #check which class and do further works                    
                    predict = Vote_p(cp, w_vector)        
                    if predict == c:
                        ws[k] += 1
                    else:
                        new_p = cp + c * w_vector
                        vs.append(new_p)
                        cp = new_p
                        ws.append(1)
                        k += 1

    return vs,ws

def Vote_p(w, x):    
    if np.dot(w,x) > 0 :
        return 1
    else:
        return -1

def result_show(groundtrain,predict):
    t = []
    p = []
    with open(groundtrain, "r+") as f:
        for line in f:
            t.append( int(line) )

    with open(predict, "r+") as f:
        for line in f:
            p.append( int(line) )
    Final_score(t,p)


def Final_score(Pre_value,test_label):
    count_a=0.0;
    count_b=0.0;
    count_c=0.0;
    count_d=0.0;
   
    for i in range(0,len(test_label)-1):
        if test_label[i]==0:
           test_label[i]=-1; 
    for i in range(0,len( Pre_value)-1):
        if  Pre_value[i]==1:
            if Pre_value[i]==test_label[i]:
               count_c=count_c+1;
            else:
               count_d=count_d+1;
        else:
            if Pre_value[i]==test_label[i]:
               count_a=count_a+1;
            else:
               count_b=count_b+1;
    precision=(count_d)/(count_b+count_d);
    recall=(count_d)/(count_c+count_d);
    F_mearure=2*(precision*recall)/(precision+recall);
    accuracy=(count_a+count_d)/(count_a+count_b+count_c+count_d);
    final_score=50*accuracy+50*F_mearure;
    return final_score, accuracy


# Xtrain_file = "Xtrain.csv"
# Ytrain_file = "Ytrain.csv"
# test_data_file = "test_data.csv" 
# label_file = "y_label.csv"
# pred_file =  "pred.csv"


# Ratio_init=np.array([0.01,0.02,0.05,0.1,0.2,1.0]);
# train_size = 450;
# for i in range(len(Ratio_init)):    
#     run(Xtrain_file,Ytrain_file,test_data_file,pred_file)
#     result_show( label_file, pred_file )


