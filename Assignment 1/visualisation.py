import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import os

sns.set_style("darkgrid")

def load_data(file_name):
	data = pd.read_csv(file_name)
	data.head()

	return data

def print_data_stats(data):
	number_of_rows = len(data)
	number_of_authorised = len(data[data['simple_journal'] == 'Settled'])
	number_of_fraud = len(data[data['simple_journal'] == 'Chargeback'])
	number_of_refused = len(data[data['simple_journal'] == 'Refused'])
	print(f"Number of transactions: {number_of_rows} \nAuthorised: {number_of_authorised} \nFraud: {number_of_fraud} \n(Refused: {number_of_refused})")

def pre_process(data):
	# The class label needs to be determined. 
	# We can neglect the refused cases since we don't know if it was fraud or not.
	# We assign class 1 for Chargeback and class 0 for Settled
	pdata = data.loc[~(data["simple_journal"] == "Refused")]
	pdata.loc[pdata["simple_journal"] == "Settled", "simple_journal"] = 0
	pdata.loc[pdata["simple_journal"] == "Chargeback", "simple_journal"] = 1

	# To prepare the data to run a model on it we have to convert the non-number
	# columns into (machine-understandable) numerical values.
	# Relevant columns (possibly) include the mail_id, ip_id, card_id, shopperinteraction.
	le = LabelEncoder()
	pdata['mail_id'] = le.fit_transform(pdata['mail_id'])
	pdata['ip_id'] = le.fit_transform(pdata['ip_id'])
	pdata['card_id'] = le.fit_transform(pdata['card_id'])

	# The creationdata currently also contains a time, which is not necessarily needed
	# for the visualization. So we create a new column containing only the date.
	pdata['date'] = pd.to_datetime(pdata['creationdate']).dt.date

	return pdata

# Split and save the data in fraud and non fraud cases
def get_fraud_data(data):
	return data.loc[data["simple_journal"] == 1]

def get_nonfraud_data(data):
	return data.loc[data["simple_journal"] == 0]

def get_bar_plot(data, feature, bundle_small):
	pdata = data.groupby(['simple_journal'])[feature].value_counts(normalize=True).rename('ratio').reset_index()
	pdata.set_index(feature)

	if bundle_small > 0:
		pdata_1p = pdata.loc[pdata["ratio"] <= bundle_small]
		pdata_1p = pdata_1p.groupby(['simple_journal']).sum().reset_index()
		pdata_1p[feature] = 'Others'

		pdata = pdata.loc[pdata["ratio"] > bundle_small]
		pdata = pdata.append(pdata_1p)

	f, (ax) = plt.subplots(1,figsize =(10,4))
	ax = sns.barplot(x=feature, y='ratio', hue='simple_journal', data=pdata)
	ax.set(ylabel = "Percentage of class")
	ax.set(xlabel = feature)
	h, l = ax.get_legend_handles_labels()
	ax.legend(h, ["Non-Fraud","Fraud"], title="Classes", loc='upper right')
	plt.savefig('figures/'+feature+'.png')
	plt.show()

	


