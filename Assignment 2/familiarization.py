import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("darkgrid")

def load_data(file_name):
	data = pd.read_csv(file_name)
	data.head()

	return data

def pre_process(data):
	# Preprocess the training data set
	# Slice the datetime col to a separate date and time of day col
	# Drop original datetime column
	data["date"] = data.DATETIME.str[:-2]
	data["hour"] = data.DATETIME.str[9:].astype(str).astype(int)
	data.drop(columns=["DATETIME"],axis=1,inplace=True)

	return data

def plot_signal(data,col,start,end, title,color):
	f,ax = plt.subplots(figsize=(22,3))
	f.suptitle(title)
	ax = sns.lineplot(data=data.loc[start:end,col],color=color)
	ax.set_xticks(range(start,end+1,24))
	plt.show()	

def plot_correlation(data):
	tdata = data.drop(columns=["date","hour","ATT_FLAG"],axis=1)

	correlation = tdata.corr()

	for col in correlation.columns:
		correlation[col] = correlation[col].fillna(0)

	f, ax = plt.subplots(figsize=(25,20))
	ax = sns.heatmap(correlation)
	ax.set_title('Correlation between signals')
	plt.show()
	






	


