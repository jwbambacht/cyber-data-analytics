import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import comparison as comp
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

sns.set_style("darkgrid")

def load_data(file_name):
	data = pd.read_csv(file_name)
	data.head()

	return data

def get_anomalous_regions(method, plot=False):

	if method == "pca":
		predicted = comp.load_data("data/pca_data.csv")
		data = comp.load_data("data/tdata2.csv")
		labels = comp.load_data("data/labels2.csv")
		start_attacks = [1727, 2027, 2337, 2827, 3497, 3727, 3927]
		end_attacks = [1776, 2050, 2396, 2920, 3556, 3820, 4036]
		threshold = 0.5010854719373081
	else:
		predicted = comp.load_data("data/arma_data.csv")
		data = comp.load_data("data/data_test.csv")
		labels = comp.load_data("data/labels_test.csv")
		start_attacks = [298, 633, 868, 938, 1230, 1575, 1941]
		end_attacks = [367, 697, 898, 968, 1329, 1654, 1970]

	signals = predicted.columns
	anomalous_regions = list()
	predicted_labels = list()

	for signal in signals:
		predicted_signal = predicted[signal]
		training_data_signal = data[signal]
		residual = abs(training_data_signal-predicted_signal)
		residual = residual/max(residual)

		if method == "arma":
			threshold = np.mean(residual)+3*np.std(residual)

		ranges = list()
		for i in range(len(start_attacks)):
			ranges.append(range(start_attacks[i],end_attacks[i]))

		if plot == True:
			fig,ax = plt.subplots(figsize=(22,3))
			plt.plot(residual, color='blue')
			plt.plot(labels, color='yellow')
			plt.axhline(threshold, color='green')

		labels_pred = np.zeros(len(residual))

		anomalous_region = 0

		for i in range(len(residual)):
			if residual[i] > threshold:
				labels_pred[i] = 1
				for r in ranges:
					if i in r:
						anomalous_region += 1
						ranges.remove(r)
						continue;

		anomalous_regions.append(anomalous_region)
		predicted_labels.append(labels_pred)

	return anomalous_regions, labels, predicted_labels

def performance(labels, labels_pred, print=False):
	tn, fp, fn, tp = confusion_matrix(labels, labels_pred).ravel()
	accuracy = accuracy_score(labels,labels_pred)
	precision = precision_score(labels,labels_pred)
	recall = recall_score(labels,labels_pred)

	if print == True:
		print("Accuracy:", accuracy)
		print("Precision:", precision)
		print("Recall:", recall)
		print("TP:", tp)
		print("FP:", fp)
		print("FN:", fn)
		print("TN:", tn)

	return accuracy, precision, recall, tp, fp, fn, tn







	






	


