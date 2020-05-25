import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from math import sqrt

from statsmodels.graphics import tsaplots
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import arma_helper

sns.set_style("darkgrid")

def load_data(file_name):
	data = pd.read_csv(file_name)
	data.head()

	return data

# Preprocess the data set
def pre_process(data, test=False):

	tdata = data.copy()

	del tdata['DATETIME']

	# Remove columns with binary values, or if mean or variance is zero
	for col in tdata.columns:
		if col != 'ATT_FLAG':
			if col.startswith("S_") or np.mean(tdata[col]) == 0 or np.var(tdata[col]) == 0:
				del tdata[col]

	# Create attack labels for test set
	if test == True:
		tdata["ATT_FLAG"] = 0
		start_attack = [298, 633, 868, 938, 1230, 1575, 1941]
		end_attack = [367, 697, 898, 968, 1329, 1654, 1970]

		for i in range(7):
			x = np.arange(start_attack[i],end_attack[i]+1,1)
			for row in x:
				tdata.at[row,"ATT_FLAG"] = 1

	tdata = tdata.rename(columns = {' ATT_FLAG':'ATT_FLAG'})
	labels = tdata["ATT_FLAG"]
	
	tdata.drop(columns=["ATT_FLAG"],axis=1,inplace=True)

	nan_columns = list()
	for col in tdata.columns:
		if np.isnan(np.mean(tdata[col])) or np.isnan(np.var(tdata[col])):
			nan_columns.append(col)

	tdata.drop(columns=nan_columns,axis=1,inplace=True)

	return tdata, labels

def compute_parameters(compute, data, sensors, signals_tank):
	# Pre-calculated orders:
	# L_T1 7 7 -16891.57666656138
	# L_T2 7 7 -1937.300859995401
	# L_T3 9 9 -10566.014270195785
	# L_T4 3 7 7869.872628503199
	# L_T5 8 9 -1182.7290762813573
	# L_T6 9 8 -13160.396504732827
	# L_T7 4 7 13939.371246562932
	# F_V2 9 9 77325.15483794194
	# F_PU2 4 8 75094.18221067122
	# P_J306 8 9 56947.17608324569
	if compute_parameters == True:
		p = [1]*len(sensors)
		q = [0]*len(sensors)
		aic = [float('inf')]*len(sensors)
		params_AR = range(1,10)
		params_MA = range(0,10)

		for sensor in sensors:
			index = sensors.index(sensor)
			for i in params_AR:
				for j in params_MA:
					try:
						aic_score = arma_helper.compute_aic(data[sensor],i,j)
						if aic_score < aic[index]:
							aic[index] = aic_score
							p[index] = i
							q[index] = j
					except: 
						continue;

			print(sensor, p[index], q[index], aic[index])
	else:
		if signals_tank == True:
			# Pre calculated order tank signals: ["L_T1","L_T2","L_T3","L_T4","L_T5","L_T6","L_T7"]
			p = [7, 7, 9, 3, 8, 9, 4]
			q = [7, 7, 9, 7, 9, 8, 7]
		else:
			# Pre-calculated orders from each category one signals: ["L_T2","F_PU2","F_V2","P_J306"]
			p = [7, 4, 9, 8]
			q = [7, 8, 9, 9]

	return p, q

def compute_aic(data,p,q):
	arma_model = ARIMA(data, order=(p,0,q));
	arma_fit = arma_model.fit(disp=0);
	
	return arma_fit.aic

	
def plot_signals(data):
	for col in data.columns:
		fig, ax = plt.subplots(figsize=(22,4))
		ax = plt.plot(data[col])
		plt.show()

def predict(coef, history):
	yhat = 0.0
	for i in range(1, len(coef)+1):
		yhat += coef[i-1] * history[-i]
	return yhat

def difference(dataset):
	diff = list()
	for i in range(1, len(dataset)):
		value = dataset[i] - dataset[i - 1]
		diff.append(value)
	return np.array(diff)

def plot_autocorrelation(data, sensor):
    print("Sensor "+sensor)
    fig, ax = plt.subplots(figsize=(22,4))
    tsaplots.plot_acf(data, lags=25, ax=ax)
    ax.set_title("Autocorrelation of sensor "+sensor, size=16)
    plt.show()
    fig, ax = plt.subplots(figsize=(22,4))
    tsaplots.plot_pacf(data, lags=50, ax=ax)
    ax.set_title("Partial Autocorrelation of sensor "+sensor, size=16)
    plt.show()

def predict_signal(sensor, train_data, test_data, p, q, plot_prediction):
    history = [x for x in train_data]
    predictions = list()

    # Train and fit the model using the parameters and train data, compute the residual and compute the AR
    # and MA coefficients, required for the prediction of the next point
    model = ARIMA(history, order=(p, 0, q))
    model_fit = model.fit()
    resid = model_fit.resid
    ar_coef, ma_coef = model_fit.arparams, model_fit.maparams

    # For each sensor value in the test set we apply differencing on the sensor value and predict the next point
    # We add this predicted value to the set of predictions
    for i in range(len(test_data)):
        difference = arma_helper.difference(history)
        y_hat = history[-1]+arma_helper.predict(ar_coef, difference) + arma_helper.predict(ma_coef, resid)
        predictions.append(y_hat)
        observation = test_data.loc[i]
        history.append(observation)
    
    # Calculate RMSE
    rmse = sqrt(mean_squared_error(test_data, predictions))

    print("RMSE of sensor "+sensor, "{:.3f}".format(rmse))
    
    if plot_prediction == True:
	    # Plot the original and predicted signals for each sensor
	    fig,ax = plt.subplots(figsize=(22,4))
	    plt.plot(test_data)
	    plt.plot(predictions, color='red')
	    ax.set_title('Signal '+sensor+' and predicted signal with RMSE='+"{:.3f}".format(rmse),size=16)
	    ax.legend(loc='upper right', labels=['Original signal','Predicted signal'])
	    plt.show()

    return predictions

def get_residual(test_set, predictions):
	residual = abs(test_set-predictions)
	residual = residual/max(residual)

	return residual

def plot_residual_anomalies(test_set, predictions, test_labels, sensor):
	# Determine residual based on the test set points and predictions
	# The threshold equals the mean plus three times the standard deviation (99.7% of all points)
	# Points that are above this threshold are labeled as an anomalie
	residual = arma_helper.get_residual(test_set, predictions)
	threshold = np.mean(residual)+3*np.std(residual)
	anomalies = [(x > threshold) for x in residual]

	# Plot the true labels, anomalies, residual error and threshold in one figure
	fig,ax = plt.subplots(figsize=(22,4))
	plt.fill_between(range(len(test_labels)), 0, test_labels, color='yellow')
	plt.plot(anomalies, color='magenta')
	plt.plot(residual, color='blue')
	plt.axhline(y=threshold, color='green')
	ax.set_title("Residual Plot with predicted anomalies and true attacks for signal "+sensor,size=16)
	ax.legend(loc='upper right', labels=['Predicted Anomalies','Residual error','Threshold','True attacks'])
	ax.set_ylabel('Normalized residual error',size=14)
	plt.show()

	# Determine the evaluation metrics to see the performance
	arma_helper.print_performance(test_labels, anomalies)

def print_performance(test_labels, anomalies):
    tn, fp, fn, tp = confusion_matrix(test_labels, anomalies).ravel()
    print('TP', tp, 'FP', fp, 'FN', fn, 'TN', tn)
    print('Accuracy', accuracy_score(test_labels, anomalies))
    print('Precision', precision_score(test_labels, anomalies))
    print('Recall', recall_score(test_labels, anomalies))
    print('F1', f1_score(test_labels,anomalies))
    print('ROC', roc_auc_score(test_labels,anomalies))


