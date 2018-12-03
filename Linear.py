import numpy as np
from numpy.linalg import inv
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt
from sklearn import preprocessing

### This is the class that does the job
class LM:
	def __init__(self,l):
		self.l = l
	def fit(self, X,Y):
		### Solve for the parameter of a linear model (no need of intercept because data are standardized)
		return(inv(X.transpose().dot(X) + self.l*np.identity(np.shape(X)[1])).dot(X.transpose()).dot(Y))
	def predict(self, X, parameters):
		### Predict with the fitted parameters
		return(np.dot(X,parameters))
	def RMSE(self, X_true, X_predicted):
		### Compute the RMSE
		return(np.sqrt(np.mean((X_true-X_predicted)**2)))

def CV(object, predictors, target, regularization_path, realization):
	rmse = []
	for lmb in regularization_path:
		r = LM(lmb)
		error = []
		for k in range(realization):
			X_train, X_val, y_train, y_val = train_test_split(predictors, target, test_size=0.2)
			para = r.fit(X_train, y_train)
			prediction = r.predict(X_val, para)
			error.append(r.RMSE(y_val,prediction))
		rmse.append(np.mean(error))
	#### Get the regularization parameter that minimize the rmse
	optimum_lmb = regularization_path[np.argmin(rmse)]
	#### Get the minimum RMSE
	min_val_error = min(rmse)
	return(optimum_lmb, rmse, min_val_error)

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
			best_lmb, err, validation_error = CV(LM, RXTrain, YTrain, reg_path, 50)
			val_err.append(validation_error)
			sbs.append(subset)
		idx = np.argmin(val_err)
		ftr.append(sbs[idx])
		feature_selection_error.append(min(val_err))
	best_feature = ftr[np.argmin(feature_selection_error)]
	return(best_feature)


def output_linear_model(XTrain, XTest, YTrain,YTest, plot = False, AllFeatures = False):
	#### Standardize the variables but preserve the information so that you can use it later to report the error in the true space
	scaler_ts_training_X = preprocessing.StandardScaler().fit(XTrain)
	scaler_ts_training_Y = preprocessing.StandardScaler().fit(YTrain.reshape(-1,1))
	XTrain = preprocessing.scale(XTrain)
	YTrain = preprocessing.scale(YTrain.reshape(-1,1))
	### Define the regularization path
	reg_path = np.logspace(-3, 0, num=10)
	#### Greedy search for features
	if AllFeatures == False:
		best_feature = variable_selection(XTrain, YTrain, reg_path)
		XTrain = XTrain[:,best_feature]
	### Run cross validation
	best_lmb, err, validation_error = CV(LM, XTrain, YTrain, reg_path, 50)
	### Training error
	r = LM(best_lmb)
	para = r.fit(XTrain,YTrain)
	YTrain_pred = r.predict(XTrain, para)
	YTrain_pred = scaler_ts_training_Y.inverse_transform(YTrain_pred.reshape(1,-1))
	YTrain = scaler_ts_training_Y.inverse_transform(YTrain.reshape(1,-1))
	training_error = r.RMSE(YTrain,YTrain_pred)
	### Test set
	XTest = scaler_ts_training_X.transform(XTest)
	if AllFeatures == False:
		XTest = XTest[:,best_feature]
	Ypred = r.predict(XTest, para)
	Ypred = scaler_ts_training_Y.inverse_transform(Ypred.reshape(1,-1))
	test_error = r.RMSE(YTest,Ypred)
	print('training error:', training_error)
	print('test error:', test_error)
	if plot == True:
		plt.plot(reg_path, err, label = 'Regularization path')
		plt.scatter(best_lmb, min(err), marker = '*', s = 80, c = 'red', label = r'Best $\lambda$')
		plt.xlabel(r'$\lambda$', fontsize = 14)
		plt.ylabel('RMSE', fontsize = 14)
		plt.legend(loc = 'best', fontsize = 16)
		plt.show()
	return(training_error,test_error)
if __name__ == '__main__':
	np.random.seed(5)
	tmp = np.loadtxt('Input/synthetic_data.txt')
	df = tmp[0:100,1:(np.shape(tmp)[1])]
	target = tmp[0:100,0]
	###
	XTrain, XTest, YTrain, YTest = train_test_split(df, target, test_size=0.2)
	output_linear_model(XTrain, XTest, YTrain,YTest)
