import numpy as np
from numpy.linalg import inv
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt
from scipy.spatial.distance import pdist, squareform, cdist
import sys
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import ParameterGrid

### Here I use Reproducing kernel Hilber spaces (RKHS)
### As an example I use a Gaussian kernel but this can be easily changed in the main class
class KernelRegression:
	def __init__(self,l,theta):
		self.l = l
		self.theta = theta
	def make_kernel(self, X, Y):
		pairwise_dists = cdist(X, Y, 'sqeuclidean')
		K = np.exp(-0.5*pairwise_dists / self.theta**2)
		return(K)
	def fit(self, X, Y, W):
		### Solve for the parameter of a linear model
		return(inv(W + self.l*np.identity(np.shape(X)[0])).dot(Y))
	def predict(self, Xt, Xv, parameters):
		### Predict with the fitted parameters
		k = self.make_kernel(Xt,Xv)
		return(k.transpose().dot(parameters))
	def RMSE(self, X_true, X_predicted):
		### Compute the RMSE
		return(np.sqrt(np.mean((X_true-X_predicted)**2)))

### Run cross validation
def CV(object, predictors, target, regularization_path, realization):
	rmse = []
	for hyper in range(len(regularization_path)):
		r = KernelRegression(regularization_path[hyper]['lambda'], regularization_path[hyper]['theta'])
		error = []
		for k in range(realization):
			X_train, X_val, y_train, y_val = train_test_split(predictors, target, test_size=0.2)
			kernel = r.make_kernel(X_train, X_train)
			para = r.fit(X_train, y_train, kernel)
			prediction = r.predict(X_train, X_val, para)
			error.append(r.RMSE(y_val,prediction))
		rmse.append(np.mean(error))
	#### Get the regularization parameter that minimize the rmse
	optimum_lmb = regularization_path[np.argmin(rmse)]['lambda']
	optimum_tht = regularization_path[np.argmin(rmse)]['theta']
	#### Get the minimum RMSE
	min_val_error = min(rmse)
	return(optimum_lmb, optimum_tht, rmse, min_val_error)


def variable_selection(x_Train, y_Train, reg_path):
	ftr = []
	feature_selection_error = []
	features = [j for j in range(x_Train.shape[1])]
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
			R_x_Train = x_Train[:,subset]
			### Run cross validation
			best_lmb, best_tht, err, validation_error = CV(KernelRegression, R_x_Train, y_Train, reg_path, 20)
			val_err.append(validation_error)
			sbs.append(subset)
		idx = np.argmin(val_err)
		ftr.append(sbs[idx])
		feature_selection_error.append(min(val_err))
	best_predictors = ftr[np.argmin(feature_selection_error)]
	return(best_predictors)

def output_kernel_model(XTrain, XTest, YTrain,YTest, AllFeatures = False):
	#### Standardize the variables but preserve the information so that you can use it later to report the error in the true space
	scaler_ts_training_X = preprocessing.StandardScaler().fit(XTrain)
	scaler_ts_training_Y = preprocessing.StandardScaler().fit(YTrain.reshape(-1,1))
	XTrain = preprocessing.scale(XTrain)
	YTrain = preprocessing.scale(YTrain.reshape(-1,1))
	### Define the regularization path
	reg_path = {'lambda': np.logspace(-3, 0, num=10), 'theta': np.logspace(0,1.5,num = 15)}
	reg_path = list(ParameterGrid(reg_path))
	#### Greedy search for features
	if AllFeatures == False:
		best_feature = variable_selection(XTrain, YTrain, reg_path)
		XTrain = XTrain[:,best_feature]
	else:
		best_feature = [j for j in range(XTrain.shape[1])]
	best_lmb, best_tht, err, validation_error = CV(KernelRegression, XTrain, YTrain, reg_path, 50)
	### Training error
	r = KernelRegression(best_lmb, best_tht)
	kernel = r.make_kernel(XTrain, XTrain)
	para = r.fit(XTrain, YTrain, kernel)
	YTrain_pred = r.predict(XTrain, XTrain, para)
	YTrain_pred = scaler_ts_training_Y.inverse_transform(YTrain_pred.reshape(1,-1))
	YTrain = scaler_ts_training_Y.inverse_transform(YTrain.reshape(1,-1))
	training_error = r.RMSE(YTrain,YTrain_pred)
	### Test set
	XTest = scaler_ts_training_X.transform(XTest)
	if AllFeatures == False:
		XTest = XTest[:,best_feature]
	Ypred = r.predict(XTrain, XTest, para)
	Ypred = scaler_ts_training_Y.inverse_transform(Ypred.reshape(1,-1))
	test_error = r.RMSE(YTest,Ypred)
	### Print results
	print('training error:', training_error)
	print('test error:', test_error)
	return(training_error,test_error, best_feature)

if __name__ == '__main__':
	np.random.seed(5)
	tmp = np.loadtxt('Input/synthetic_data.txt')
	df = tmp[0:100,1:(np.shape(tmp)[1])]
	target = tmp[0:100,0]
	###
	XTrain, XTest, YTrain, YTest = train_test_split(df, target, test_size=0.2)
	train_error, tst_error, predictors = output_kernel_model(XTrain, XTest, YTrain,YTest)
	print(predictors)
