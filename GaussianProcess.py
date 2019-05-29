#%%
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, DotProduct, Sum, ConstantKernel
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import ParameterGrid
#%%
class GaussProc:
	def __init__(self, krn):
		self.krn = krn
	def fit(self, X,Y):
		gp = GaussianProcessRegressor(kernel=self.krn, optimizer = None, n_restarts_optimizer=0)
		gpr = gp.fit(X, Y)
		### Fit the data
		fitted = gp.predict(X)
		return(gpr, fitted)
	def predict(self, gpr, X):
		prd, sigma = gpr.predict(X, return_std=True)
		return(prd,sigma)
	def RMSE(self, X_true, X_predicted):
		return(np.sqrt(np.mean((X_true-X_predicted)**2)))

### Run cross validation
def CV(object, predictors, target, regularization_path, realization):
	rmse = []
	for hyper in range(len(regularization_path)):
		r = GaussProc(regularization_path[hyper ]['kernel'](regularization_path[hyper ]['l']))
		error = []
		for k in range(realization):
			X_train, X_val, y_train, y_val = train_test_split(predictors, target, test_size=0.2)
			gpr, _ = r.fit(X_train, y_train)
			prd, _ = r.predict(gpr,X_val)
			error.append(r.RMSE(y_val,prd))
		rmse.append(np.mean(error))
	#### Get the regularization parameter that minimize the rmse
	optimum_krn = regularization_path[np.argmin(rmse)]['kernel']
	optimum_L = regularization_path[np.argmin(rmse)]['l']
	#### Get the minimum RMSE
	min_val_error = min(rmse)
	return(optimum_krn, optimum_L, rmse, min_val_error)


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
			_, _, err, validation_error = CV(GaussProc, R_x_Train, y_Train, reg_path, 20)
			val_err.append(validation_error)
			sbs.append(subset)
		idx = np.argmin(val_err)
		ftr.append(sbs[idx])
		feature_selection_error.append(min(val_err))
	best_predictors = ftr[np.argmin(feature_selection_error)]
	return(best_predictors)

def output_GP(XTrain, XTest, YTrain,YTest, AllFeatures = False):
	#### Standardize the variables but preserve the information so that you can use it later to report the error in the true space
	scaler_ts_training_X = preprocessing.StandardScaler().fit(XTrain)
	scaler_ts_training_Y = preprocessing.StandardScaler().fit(YTrain.reshape(-1,1))
	XTrain = preprocessing.scale(XTrain)
	YTrain = preprocessing.scale(YTrain.reshape(-1,1))
	### Define the regularization path
	reg_path = {'kernel': [Matern], 'l': np.logspace(0,1,15)}
	reg_path = list(ParameterGrid(reg_path))
	#### Greedy search for features
	if AllFeatures == False:
		best_feature = variable_selection(XTrain, YTrain, reg_path)
		XTrain = XTrain[:,best_feature]
	else:
		best_feature = [j for j in range(XTrain.shape[1])]
	best_krn, best_l, err, validation_error = CV(GaussProc, XTrain, YTrain, reg_path, 50)
	### Training error
	r = GaussProc(best_krn(best_l))
	gpr, _ = r.fit(XTrain, YTrain)
	YTrain_pred , _ = r.predict(gpr,XTrain)
	YTrain_pred = scaler_ts_training_Y.inverse_transform(YTrain_pred.reshape(1,-1))
	YTrain = scaler_ts_training_Y.inverse_transform(YTrain.reshape(1,-1))
	training_error = r.RMSE(YTrain,YTrain_pred)
	### Test set
	XTest = scaler_ts_training_X.transform(XTest)
	if AllFeatures == False:
		XTest = XTest[:,best_feature]
	Ypred, sigma_pred = r.predict(gpr,XTest)
	Ypred = scaler_ts_training_Y.inverse_transform(Ypred.reshape(1,-1))
	test_error = r.RMSE(YTest,Ypred)
	### Print results
	print('training error:', training_error)
	print('test error:', test_error)
	return(training_error,test_error, best_feature, np.squeeze(Ypred), sigma_pred)

#%%
if __name__ == '__main__':
	np.random.seed(5)
	tmp = np.loadtxt('Input/synthetic_data.txt')
	df = tmp[0:100,1:(np.shape(tmp)[1])]
	target = tmp[0:100,0]
	###
	XTrain, XTest, YTrain, YTest = train_test_split(df, target, test_size=0.2)
	train_error, tst_error, predictors, Y_pred, sigma_pred = output_GP(XTrain, XTest, YTrain,YTest, AllFeatures =False)

	plt.scatter(YTest,Y_pred)