import numpy as np
import math
import csv 

def run(Xtrain_file, Ytrain_file, test_data_file, pred_file):
    # your implementation here
    # read data from Xtrain_file, Ytrain_file and test_data_file
    # your algorithm
    # save your predictions into the file pred_file
    counter = 0
    alpha = 0.2
    #open files and train

    v_size,W_Cs,Class_sum,W_sums = train(Xtrain_file,Ytrain_file)
    
    total_classes = sum(Class_sum.values())
    W_sum = sum(W_sums.values())
    counter = 0

    #predict on testing data set
    with open(test_data_file, "r+") as test_f:
        with open(pred_file, "w+") as pred_f:
            for lines in test_f:
                counter += 1 

                max_p = 1
                splited_line = lines.split(",")
                w_vector = np.array([int(x) for x in splited_line])
                
                prediction = None
                max_p = float('-inf')
                for classes, count in Class_sum.items():

                    assert classes==0 or classes==1
                    p = math.log(float(count)/total_classes)

                    for i, num in enumerate(w_vector):
                        p +=  num * math.log( (float(W_Cs[classes][i])+alpha) / (W_sums[classes] + alpha*v_size))
                    if p > max_p:
                        max_p = p
                        prediction = classes
                pred_f.write( str(prediction) + "\n" )


def train(Xtrain,Ytrain):
    v_size = None
    W_Cs = {}
    Class_sum = {}
    W_sums = {}
    counter = 0
    with open(Xtrain, "r+") as x_f:
        with open(Ytrain, "r+") as y_f:
            for lines in x_f:
                counter += 1
                #Read data to array
                splited_line = lines.split(",")
                w_vector = np.array([int(x) for x in splited_line])
                #initialize
                if v_size == None:
                    v_size = len(w_vector)

                #train using labels 
                classes = int(y_f.readline())

                if not classes in W_Cs:
                    W_Cs[classes] = np.zeros( v_size )
                    Class_sum[classes] = 0
                    W_sums[classes] = 0

                W_Cs[classes] += w_vector
                Class_sum[classes] += 1                    
                W_sums[classes] += sum(w_vector)
    
    return v_size,W_Cs,Class_sum,W_sums


def result_show(groundtruth, predict):
    t = []
    p = []
    with open(groundtruth, "r+") as f:
        for line in f:
            t.append( int(line) )

    with open(predict, "r+") as f:
        for line in f:
            p.append( int(line) )
    Final_score(t,p)

def accuracy( truth, predict ):
    t = np.array(truth)
    p = np.array(predict)
    assert len(t) == len(p)
    s = 0

    for i in range(len(t)):
        if t[i] == p[i]:
            s += 1 

    return float(s)/len(t)

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
	F_mearure=2.0*(precision*recall*1.0)/(precision+recall);
	accuracy=(count_a+count_d)/(count_a+count_b+count_c+count_d);
	final_score=50*accuracy+50*F_mearure;
	return final_score, accuracy

# define other functions here

#Xtrain_file = "Xtrain.csv"
#Ytrain_file = "Ytrain.csv"
#test_data_file = Xtrain_file
#truth = "Ytrain.csv"

#pred_file = "Xtrain.csv"
# train_size = 450
# percent = [0.01, 0.02, 0.05, 0.1, 0.2, 1]
# for i in range(len(percent)):
#     print("percentage: %d%%"%(percent[i]*100))
#     run(Xtrain_file,Ytrain_file,test_data_file,pred_file)
#     report( label_file, pred_file )
