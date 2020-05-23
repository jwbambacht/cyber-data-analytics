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

sns.set_style("darkgrid")

def load_data(file_name):
	data = pd.read_csv(file_name)
	data.head()

	return data

# Preprocess the data set
def pre_process(data):

	# Remove the datetime and truth label columns from dataset
	tdata = data.copy()

	tdata = tdata.rename(columns = {' ATT_FLAG':'ATT_FLAG'})

	labels = tdata["ATT_FLAG"]

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
	fig, ax = plt.subplots(figsize=(10,5))
	x = np.arange(1, n_features+1, step=1)
	y = np.cumsum(pca.explained_variance_ratio_)

	plt.ylim(0.18,1.1)
	plt.plot(x, y, marker='o', linestyle='--', color='b')

	plt.xlabel('# Components', fontsize=14)
	plt.xticks(np.arange(0, n_features+1, step=1))
	plt.ylabel('Cumulative variance (%)', fontsize=14)
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





	






	


