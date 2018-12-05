import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas
from sklearn.model_selection import ParameterGrid


def CV(XTrain,YTrain, para, num_iterations):
	val_iterations = []
	for hyper in range(len(para)):
		err = []
		for realization in range(num_iterations):
			X_train, X_val, y_train, y_val = train_test_split(XTrain, YTrain, test_size=0.2)
			regr_1 = RandomForestRegressor(max_depth = para[hyper]['depth'], min_samples_split = para[hyper]['split'], min_samples_leaf = para[hyper]['leaf'])
			regr_1.fit(X_train,y_train)
			y_pred = regr_1.predict(X_val)
			err.append(np.sqrt(np.mean((y_val - y_pred)**2)))
		val_iterations.append(np.mean(err))
	val_error = min(val_iterations)
	return(val_iterations, val_error)


def output_regression_forest(XTrain, YTrain, XTest, YTest):
	param_grid = {'depth': [2, 4, 8, 16], 'split': [2, 3, 5, 8, 10, 12], 'leaf': [1, 3, 5, 8,  10, 12]}
	reg_path = list(ParameterGrid(param_grid))
	err, val_err = CV(XTrain, YTrain, reg_path, 5)
	idx = np.argmin(err)
	regr_tree = RandomForestRegressor(max_depth = reg_path[idx]['depth'], min_samples_split = reg_path[idx]['split'], min_samples_leaf = reg_path[idx]['leaf'])
	regr_tree.fit(XTrain,YTrain)
	y_pred = regr_tree.predict(XTrain)
	train_error = (np.sqrt(np.mean((YTrain - y_pred)**2)))
	# Predict
	y_1 = regr_tree.predict(XTest)
	test_err = (np.sqrt(((YTest - y_1)**2).mean()))
	print('Training error:', train_error)
	print('Test error:', test_err)

	return(train_error, test_err)



if __name__ == '__main__':
	np.random.seed(5)
	tmp = np.loadtxt('Input/synthetic_data.txt')
	length_to_take = 300
	df = tmp[0:length_to_take,1:np.shape(tmp)[1]]
	target = tmp[0:length_to_take,0]
	XTrain, XTest, YTrain, YTest = train_test_split(df, target, test_size=0.1)
	train_err, test_err = output_regression_forest(XTrain, YTrain, XTest, YTest)

