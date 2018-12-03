import numpy as np

def output_constant_predictor(YTrain, YTest):
	naive_training_predictions = np.repeat(np.mean(YTrain), np.shape(YTrain)[0])
	naive_test_predictions = np.repeat(np.mean(YTrain), np.shape(YTest)[0])
	training_error = np.sqrt(np.mean((YTrain - naive_training_predictions)**2) )
	test_error = np.sqrt(np.mean((YTest - naive_test_predictions)**2) )
	print('Training error:', training_error)
	print('Test error:', test_error)
	return(training_error, test_error)
