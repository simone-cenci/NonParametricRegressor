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
from scipy import stats
def bootstrap(sample, num_boot,stat):
        stat_of_new_sample = []
        for j in range(num_boot):
                new_sample = np.random.choice(sample, size = len(sample))
                stat_of_new_sample.append(stat(new_sample))
        return np.std(stat_of_new_sample)

if __name__ == '__main__':
	training_sizes = [10, 20, 40, 60, 120, 240]
	train = []
	test = []
	train_error = []
	test_error = []
	for training_length in training_sizes:
		tmp = np.loadtxt('Input/synthetic_data.txt')
		length_to_take = training_length
		df = tmp[0:length_to_take,1:np.shape(tmp)[1]]
		target = tmp[0:length_to_take,0]
		### import data frame
		model = sys.argv[1]
		normalizza = False
		print('Regression Type:', model)
		num_realizations = 50
		train_it = []
		test_it = []
		seed = [int(x) for x in range(1,num_realizations+1)]
		for k in range(num_realizations):
			np.random.seed(seed[k])
			XTrain, XTest, YTrain, YTest = train_test_split(df, target, test_size=0.1)
			if model == 'linear':
				training_, testing_= output_linear_model(XTrain, XTest, YTrain, YTest)
				train_it.append(training_)
				test_it.append(testing_)
			if model == 'kernel':
				training_, testing_ = output_kernel_model(XTrain, XTest, YTrain, YTest)
				train_it.append(training_)
				test_it.append(testing_)
			if model == 'tree':
				training_, testing_ = output_regression_tree(XTrain, YTrain, XTest, YTest)
				train_it.append(training_)
				test_it.append(testing_)
			if model == 'forest':
				training_, testing_ = output_regression_forest(XTrain, YTrain, XTest, YTest)
				train_it.append(training_)
				test_it.append(testing_)
			if model == 'constant':
				training_, testing_ = output_constant_predictor(YTrain, YTest)
				train_it.append(training_)
				test_it.append(testing_)
			print('Realization number:', k, ' model:', model, 'Training size:', training_length)
		train.append(np.median(train_it))
		test.append(np.median(test_it))
		train_error.append(1.96*bootstrap(train_it, training_length*2, np.median))
		test_error.append(1.96*bootstrap(test_it, training_length*2, np.median))




	put_together = np.column_stack([training_sizes, train, train_error, test, test_error])
	file_name='Output/'+str(model)+'.txt'
	np.savetxt(file_name, put_together)
