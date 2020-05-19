import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import familiarization as fam
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
from math import sqrt

sns.set_style("darkgrid")

def load_data(file_name):
	data = pd.read_csv(file_name)
	data.head()

	return data

# Preprocess the data set
def pre_process(data):
	# Slice the datetime col to a separate date and time of day col
	# Drop original datetime column
	data["date"] = data.DATETIME.str[:-2]
	data["hour"] = data.DATETIME.str[9:].astype(str).astype(int)
	data.drop(columns=["DATETIME"],axis=1,inplace=True)

	return data

# Plot a signal
def plot_signal(data,col,start=0,end=720,title="",color="blue"):
	f,ax = plt.subplots(figsize=(22,3))
	f.suptitle(title)
	ax = sns.lineplot(data=data.loc[start:end,col],color=color)
	ax.set_xticks(range(start,end+1,24))
	plt.show()	

# Generate heatmap of all correlations between any columns
def plot_correlation(data):
	tdata = data.drop(columns=["date","hour","ATT_FLAG"],axis=1)

	correlation = tdata.corr()

	for col in correlation.columns:
		correlation[col] = correlation[col].fillna(0)

	f, ax = plt.subplots(figsize=(25,20))
	ax = sns.heatmap(correlation)
	ax.set_title('Correlation between signals')
	plt.show()

# Predict next point based on model and lag
def predict(coefficients, history):
	y_hat = coefficients[0]
	for i in range(1, len(coefficients)):
		y_hat += coefficients[i] * history[-i]
	return y_hat

# Predict signal
def predict_signal(data,col,ratio):
	tdata = data[col]
	train_size = int(len(tdata)*ratio)

	# Determine the differences from each point to its predecessor
	differences = np.array([tdata[i]-tdata[i-1] for i in range(1,len(tdata))])

	# Create a train and test set
	X_train, X_test = differences[0:train_size], differences[train_size:]

	# Train the AutoRegression model using training data and defined lag
	reg = AutoReg(X_train, lags=4)
	reg_fit = reg.fit()
	coefficients = reg_fit.params

	# Save the training points and predict new points based on the coefficients of the model and the training points
	history = [X_train[i] for i in range(len(X_train))]
	predictions = list()
	for t in range(len(X_test)):
		yhat = predict(coefficients, history)
		obs = X_test[t]
		predictions.append(yhat)
		history.append(obs)

	# Calculate RMSE
	rmse = sqrt(mean_squared_error(X_test, predictions))

	# Plot the original and predicted signal
	f,ax = plt.subplots(figsize=(22,3))
	sns.lineplot(data=X_test,color='blue', label="original")
	sns.lineplot(data=np.array(predictions),color='red', label='predicted')
	f.suptitle("Original vs Predicted signal of "+col+", RMSE: "+str(rmse))
	plt.show()	

	return rmse



	






	


