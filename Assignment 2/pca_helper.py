import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pca_helper as pca_helper
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

sns.set_style("darkgrid")

def load_data(file_name):
	data = pd.read_csv(file_name)
	data.head()

	return data

# Preprocess the data set
def pre_process(data,train_data_2=False):

	# Remove the datetime and truth label columns from dataset
	tdata = data.copy()

	tdata = tdata.rename(columns = {' ATT_FLAG':'ATT_FLAG'})

	if train_data_2 == True:
		start = [1727, 2027, 2337, 2827, 3497, 3727, 3927]
		end = [1776, 2050, 2396, 2920, 3556, 3820, 4036]

		labels = np.zeros(len(data))

		for i in range(7):
			x = np.arange(start[i],end[i]+1,1)
			for row in x:
				labels[row] = 1
	else:
		labels = np.zeros(len(data))

	tdata.drop(columns=["DATETIME","ATT_FLAG"],axis=1,inplace=True)

	# Normalize each signal by applying the zscore
	for col in tdata.columns:
		mean = np.mean(tdata[col])
		std = np.std(tdata[col])
		tdata[col] = (tdata[col]-mean)/std

	nan_columns = list()

	for col in tdata.columns:
		if np.isnan(np.mean(tdata[col])) or np.isnan(np.var(tdata[col])):
			nan_columns.append(col)

	tdata.drop(columns=nan_columns,axis=1,inplace=True)

	return tdata, labels

def number_of_components_plot(pca, n_features, threshold):
	fig, ax = plt.subplots(figsize=(10,2))
	x = np.arange(1, n_features+1, step=1)
	y = np.cumsum(pca.explained_variance_ratio_)

	plt.ylim(0.18,1.1)
	plt.plot(x, y, marker='o', linestyle='--', color='b')

	plt.xlabel('# Components', fontsize=12)
	plt.xticks(np.arange(0, n_features+1, step=1))
	plt.ylabel('Cumulative variance (%)', fontsize=12)
	plt.title('Number of components', fontsize=18)

	plt.axhline(y=0.97, color='r', linestyle='-')
	plt.text(38, 0.96, "{:.0f}%".format(97), color = 'red', fontsize=14)
	plt.axvline(x=13, color='r', linestyle='-')

	ax.grid(axis='x')
	plt.show()

	cumsum = 0
	n_components = 0
	for var in pca.explained_variance_ratio_:
		cumsum += var
		n_components += 1
		if cumsum > threshold:
			break;

	return pca.components_

def predict_labels(residual, threshold):
	labels_pred = np.zeros(len(residual))

	for i in range(len(residual)):
		if residual[i] >= threshold:
			labels_pred[i] = 1

	return labels_pred

def performance(labels, labels_pred):
	tn, fp, fn, tp = confusion_matrix(labels, labels_pred).ravel()

	print("Accuracy:", accuracy_score(labels,labels_pred))
	print("Precision:", precision_score(labels,labels_pred))
	print("Recall:", recall_score(labels,labels_pred))
	print("TP:", tp)
	print("FP:", fp)
	print("FN:", fn)
	print("TN:", tn)

def plot_threshold_search(residual, labels):
	
	thresholds = list()
	accuracies = list()
	precisions = list()
	recalls = list()
	tps = list()
	fps = list()

	mean = np.mean(residual)
	diff = (np.max(residual)-mean)/0.001

	for i in range(0,int(diff)):
		threshold = mean+i*0.001
		thresholds.append(threshold)
		labels_pred = pca_helper.predict_labels(residual,threshold)

		tn, fp, fn, tp = confusion_matrix(labels, labels_pred).ravel()

		accuracies.append(accuracy_score(labels,labels_pred))
		precisions.append(precision_score(labels,labels_pred))
		recalls.append(recall_score(labels,labels_pred))
		tps.append(tp)
		fps.append(fp)

	results = pd.DataFrame(thresholds, columns=["Threshold"])
	results['Accuracy'] = accuracies
	results['Precision'] = precisions
	results['Recall'] = recalls
	results['TP'] = tps
	results['FP'] = fps
	results['TP_NORM'] = tps/np.max(tps)
	results['FP_NORM'] = fps/np.max(fps)

	f,ax = plt.subplots(figsize=(22,5))
	plt.plot(results["Accuracy"], color='red',label='Accuracy')
	plt.plot(results["Precision"], color='blue', label='Precision')
	plt.plot(results["Recall"], color='green', label='Recall')
	plt.plot(results["TP_NORM"], color='yellow',label='Normalized TP')
	plt.plot(results["FP_NORM"], color='magenta', label='Normalized FP')
	plt.axvline(448, label='Optimal threshold')
	ax.legend(loc='upper right')
	ax.set_title("Performance metrics per threshold", size=14)
	ax.set_ylabel("Results", size=12)
	plt.show()

def plot_anomaly_range(residual, start, end,type):
	fig, ax = plt.subplots(figsize=(22,3))
	plt.plot(residual[start:end])
	ax.set_title("Part of the residual plot, which is labeled and detected to be a "+type+" anomaly",size=14)
	plt.show()





	






	


