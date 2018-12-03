import pandas
import numpy as np
import sys
from Linear import *
from Kernel import *
from RegressionTree import *
from RandomForest import *
from constant_predictor import *
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

if __name__ == '__main__':
	tmp = np.loadtxt('Input/synthetic_data.txt')
	length_to_take = 300
	df = tmp[0:length_to_take,1:np.shape(tmp)[1]]
	target = tmp[0:length_to_take,0]
	### import data frame
	model = sys.argv[1]
	normalizza = False
	train = []
	test = []
	print('Regression Type:', model)
	num_realizations = 50
	seed = [int(x) for x in range(1,num_realizations+1)]
	for k in range(num_realizations):
		np.random.seed(seed[k])
		XTrain, XTest, YTrain, YTest = train_test_split(df, target, test_size=0.1)
		if normalizza == True:
			scaler_ts_training_X = preprocessing.StandardScaler().fit(XTrain)
			scaler_ts_training_Y = preprocessing.StandardScaler().fit(YTrain.reshape(-1,1))
			XTrain = preprocessing.scale(XTrain)
			YTrain = preprocessing.scale(YTrain.reshape(-1,1))
			XTest = scaler_ts_training_X.transform(XTest)
			YTest = scaler_ts_training_Y.transform(YTest.reshape(-1,1))
		if model == 'linear':
			training_, testing_= output_linear_model(XTrain, XTest, YTrain, YTest)
			train.append(training_)
			test.append(testing_)
		if model == 'kernel':
			training_, testing_ = output_kernel_model(XTrain, XTest, YTrain, YTest)
			train.append(training_)
			test.append(testing_)
		if model == 'tree':
			training_, testing_ = output_regression_tree(XTrain, YTrain, XTest, YTest)
			train.append(training_)
			test.append(testing_)
		if model == 'forest':
			training_, testing_ = output_regression_forest(XTrain, YTrain, XTest, YTest)
			train.append(training_)
			test.append(testing_)
		if model == 'constant':
			training_, testing_ = output_constant_predictor(YTrain, YTest)
			train.append(training_)
			test.append(testing_)
	put_together = np.column_stack([train, test])
	file_name='output/'+str(model)+'_'+str(data_type)+'_'+str(length_to_take)+'.txt'
	np.savetxt(file_name, put_together)
