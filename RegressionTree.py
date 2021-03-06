#%%
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas
from sklearn.model_selection import ParameterGrid
import scipy.stats as stats
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


def CV(XTrain,YTrain, para, num_iterations):
	val_iterations = []
	for hyper in range(len(para)):
		err = []
		for realization in range(num_iterations):
			X_train, X_val, y_train, y_val = train_test_split(XTrain, YTrain, test_size=0.2)
			regr_1 = DecisionTreeRegressor(max_depth = para[hyper]['depth'], min_samples_split = para[hyper]['split'], min_samples_leaf = para[hyper]['leaf'])
			regr_1.fit(X_train,y_train)
			y_pred = regr_1.predict(X_val)
			err.append(np.sqrt(np.mean((y_val - y_pred)**2)))
		val_iterations.append(np.mean(err))
	val_error = min(val_iterations)
	return(val_iterations, val_error)

def variable_selection(XTrain, YTrain, reg_path):
	ftr = []
	feature_selection_error = []
	features = [j for j in range(XTrain.shape[1])]
	for i in range(len(features)):
		if len(ftr) != 0:
			features.remove(features[idx])
		val_err = []
		sbs = []
		for k in features:
			if len(ftr) !=0:
				subset = list(ftr[i-1] + [k])
			else:
				subset = [k]
			RXTrain = XTrain[:,subset]
			### Run cross validation
			err, validation_error = CV(RXTrain, YTrain, reg_path, 5)
			val_err.append(validation_error)
			sbs.append(subset)
		idx = np.argmin(val_err)
		ftr.append(sbs[idx])
		feature_selection_error.append(min(val_err))
	best_feature = ftr[np.argmin(feature_selection_error)]
	return(best_feature)

def output_regression_tree(XTrain, YTrain, XTest, AllFeatures = False):
	param_grid = {'depth': [2, 4, 8, 16, 32, 64, 128], 'split': [10, 30, 50, 70, 90, 100], 'leaf': [1, 5, 10, 20, 40, 80]}
	reg_path = list(ParameterGrid(param_grid))
	#### Greedy search for features
	if AllFeatures == False:
		print('Greedy feature selection ...')
		best_feature = variable_selection(XTrain, YTrain, reg_path)
		XTrain = XTrain[:,best_feature]
		print('Selected feature:', best_feature)
	err, val_err = CV(XTrain, YTrain, reg_path, 5)
	idx = np.argmin(err)
	regr_tree = DecisionTreeRegressor(max_depth = reg_path[idx]['depth'], min_samples_split = reg_path[idx]['split'], min_samples_leaf = reg_path[idx]['leaf'])
	regr_tree.fit(XTrain,YTrain)
	y_pred = regr_tree.predict(XTrain)
	train_error = (np.sqrt(np.mean((YTrain - y_pred)**2)))
	# Predict
	if AllFeatures == False:
		XTest = XTest[:,best_feature]
	Y_pred = regr_tree.predict(XTest)

	print('Training error:', train_error)


	return(train_error, Y_pred)
def compute_error_measures(YT, YP):
	rmse_ = np.sqrt(np.mean((YT-YP)**2))
	rho = stats.pearsonr(YT, YP)[0]
	R2 = rho**2
	return(rmse_, rho, R2)



if __name__ == '__main__':
	np.random.seed(5)
	tmp = np.loadtxt('Input/synthetic_data.txt')
	length_to_take = 300
	df = tmp[0:length_to_take,1:np.shape(tmp)[1]]
	target = tmp[0:length_to_take,0]
	XTrain, XTest, YTrain, YTest = train_test_split(df, target, test_size=0.2)
	train_err, Y_pred = output_regression_tree(XTrain, YTrain, XTest, AllFeatures = True)
	rmse_, rho, R2 = compute_error_measures(YTest, Y_pred)
	print('R2:', R2, \
		'\nRMSE:', rmse_)
	plt.scatter(YTest,Y_pred)