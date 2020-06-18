#%%
import numpy as np
import matplotlib.pylab as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas
from sklearn.model_selection import ParameterGrid
import scipy.stats as stats
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
from sklearn.model_selection import RandomizedSearchCV


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


def output_regression_forest(XTrain, YTrain, XTest, costumCV = False):
    if costumCV:   #### Simple cross validation
        param_grid = {'depth': [2, 4, 8, 16], 'split': [2, 3, 5, 8, 10, 12], 'leaf': [1, 3, 5, 8,  10, 12]}
        reg_path = list(ParameterGrid(param_grid))
        err, val_err = CV(XTrain, YTrain, reg_path, 5)
        idx = np.argmin(err)
        regr_tree = RandomForestRegressor(max_depth = reg_path[idx]['depth'], min_samples_split = reg_path[idx]['split'], min_samples_leaf = reg_path[idx]['leaf'])
        regr_tree.fit(XTrain,YTrain)
        y_1 = regr_tree.predict(XTest)
   
    else:       ### Random Search
        param_grid = \
        {'bootstrap': [True, False],
         'max_depth': [4, 8, 16, 32, 64, None],
         'max_features': ['auto', 'sqrt'],
         'min_samples_leaf': [1, 2, 4],
         'min_samples_split': [2, 5, 10],
         'n_estimators': [50, 100, 150, 200]}
        rf = RandomForestRegressor()
        rf_random = RandomizedSearchCV(estimator = rf, param_distributions = param_grid, \
        n_iter = 50, cv = 3, \
        verbose=0, random_state=5)      
        rf_random.fit(XTrain, YTrain)
        y_1 = rf_random.best_estimator_.predict(XTest)            

    return(y_1)



def output_classifier_forest(XTrain, YTrain, XTest):

    param_grid = \
    {'bootstrap': [True, False],
     'max_depth': [4, 8, 16, 32, 64, None],
     'max_features': ['auto', 'sqrt'],
     'min_samples_leaf': [1, 2, 4],
     'min_samples_split': [2, 5, 10],
     'n_estimators': [50, 100, 150, 200]}
    rf = RandomForestClassifier()
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = param_grid, \
    n_iter = 50, cv = 3, \
    verbose=0, random_state=5)      
    rf_random.fit(XTrain, YTrain)
    y_1 = rf_random.best_estimator_.predict(XTest)            

    return(rf_random.best_estimator_, y_1)


def compute_error_measures(YT, YP):
	rmse_ = np.sqrt(np.mean((YT-YP)**2))
	rho = stats.pearsonr(YT, YP)[0]
	R2 = rho**2
	return(rmse_, rho, R2)

#%%
if __name__ == '__main__':
    
    ### Generate a nonlinear model
    Z = np.random.normal(0,1,size = (200,3))   
    T = Z[:,0] + Z[:,1]**2/Z[:,2]
    
    ### Split data in training and testing
    XTrain, XTest, YTrain, YTest = train_test_split(Z, T, test_size=0.2)

    ## test two different cross-validation schemes
    ### Type 1 (random features selection). This is significantly faster than the next one
    Y_pred = output_regression_forest(XTrain, YTrain, XTest, costumCV = False)  
    rmse_, rho, R2 = compute_error_measures(YTest, Y_pred)
    print('R2:', R2, \
    	'\nRMSE:', rmse_,
        '\n#####')

    ### Type 2
    Y_pred = output_regression_forest(XTrain, YTrain, XTest, costumCV = True)  
    rmse_, rho, R2 = compute_error_measures(YTest, Y_pred)
    print('R2:', R2, \
    	'\nRMSE:', rmse_)
