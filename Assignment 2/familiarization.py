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

def plot_sample_signals(data):

	f,ax = plt.subplots(figsize=(22,3))
	f.suptitle('Water level of tank 1')
	ax = sns.lineplot(data=data.loc[0:720,"L_T1"],color='blue')
	ax.set_xticks(range(0,721,24))
	plt.show()

	f,ax = plt.subplots(figsize=(22,3))
	f.suptitle('Flow through pump 1')
	ax = sns.lineplot(data=data.loc[0:720,"F_PU1"],color='green')
	ax.set_xticks(range(0,721,24))
	plt.show()

	f,ax = plt.subplots(figsize=(22,3))
	f.suptitle('Status of pump 2')
	ax = sns.lineplot(data=data.loc[0:720,"S_PU2"],color='red')
	ax.set_xticks(range(0,721,24))
	ax.set_yticks((0,1))
	plt.show()

	f,ax = plt.subplots(figsize=(22,3))
	f.suptitle('Pressure of joint 280')
	ax = sns.lineplot(data=data.loc[0:720,"P_J280"],color='black')
	ax.set_xticks(range(0,721,24))
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

def plot_correlation_examples(tdata):
	f, ax = plt.subplots(figsize=(22,4))
	ax = sns.lineplot(data=tdata.loc[0:720,"F_PU2"],color='blue')
	ax.set_title('Flow in pump 2')
	plt.show()

	f, ax = plt.subplots(figsize=(22,4))
	ax = sns.lineplot(data=tdata.loc[0:720,"S_PU2"],color='red')
	ax.set_title('Status of pump 2')
	plt.show()






	


