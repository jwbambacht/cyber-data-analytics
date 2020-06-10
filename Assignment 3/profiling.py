import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import profiling as prof
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import KBinsDiscretizer

import math
import random

sns.set_style("darkgrid")

def load_data(file_name):
	data = pd.read_csv(file_name, sep=",",)
	data.head()

	return data

# Preprocess the data set
def pre_process(data, scenario):
	# Rename label column for usability, and remove background flows
	data.rename(columns={"Dur":"Duration","Proto":"Protocol","SrcAddr":"SourceAddress","Sport":"SourcePort","Dir":"Direction","DstAddr":"DestinationAddress","Dport":"DestinationPort","TotPkts":"TotalPackets","TotBytes":"TotalBytes","SrcBytes":"SourceBytes","Label(Normal:CC:Background)": "Label"}, inplace=True)
	data = data[~data.Label.str.contains("Background")]

	# Remove useless columns
	data = data.drop(columns=["sTos","dTos","State","Label"],axis=1)

	data["Date"] = data["StartTime"]

	# Create column that indicates the infected and non-infected hosts
	infected_hosts_addr = ["147.32.84.165","147.32.84.191","147.32.84.192","147.32.84.193","147.32.84.204","147.32.84.205","147.32.84.206","147.32.84.207","147.32.84.208","147.32.84.209"]
	non_infected_hosts_addr = ["147.32.84.170","147.32.84.134","147.32.84.164","147.32.87.36","147.32.80.9","147.32.87.11"]

	if scenario == 11 or scenario == 12:
		infected_hosts_addr = ["147.32.84.165","147.32.84.191","147.32.84.192"]
	
	data["Infected"] = 0
	data.loc[data["SourceAddress"].isin(infected_hosts_addr),"Infected"] = 1

	data['Protocol'] = data['Protocol'].str.upper() 

	data.sort_values(by=["StartTime"])

	return data

def encode_feature(data):
	le = LabelEncoder()
	data = le.fit_transform(data)

	return data

def discretize_feature(data, feature, nbin, strategy):
	X = np.array(data[feature]).reshape(-1,1)
	enc = KBinsDiscretizer(n_bins=nbin, encode="ordinal", strategy=strategy)
	Xt = enc.fit_transform(X)
	binsedges = enc.bin_edges_[0]

	# needs to be done otherwise the first bucket doesnt contain any data 
	binsedges[0] = binsedges[0] - 0.0000001
	new_labels = range(0, len(binsedges)-1)

	discretized = pd.cut(np.array(data[feature]), bins=binsedges, labels=new_labels)

	return discretized, binsedges

def select_infected_host(data, scenario):
	if scenario == 9 or scenario == 10:
		infected_hosts_addr = ["147.32.84.165","147.32.84.191","147.32.84.192","147.32.84.193","147.32.84.204","147.32.84.205","147.32.84.206","147.32.84.207","147.32.84.208","147.32.84.209"]
	elif scenario == 11 or scenario == 12:
		infected_hosts_addr = ["147.32.84.165","147.32.84.191","147.32.84.192"]

	counts = list()

	for host in infected_hosts_addr:
		counts.append(len(data.loc[data["SourceAddress"] == host]))

	return infected_hosts_addr[counts.index(max(counts))]

def is_infected(source, scenario):
	if scenario == 9 or scenario == 10:
		infected_hosts_addr = ["147.32.84.165","147.32.84.191","147.32.84.192","147.32.84.193","147.32.84.204","147.32.84.205","147.32.84.206","147.32.84.207","147.32.84.208","147.32.84.209"]
	elif scenario == 11 or scenario == 12:
		infected_hosts_addr = ["147.32.84.165","147.32.84.191","147.32.84.192"]

	return source in infected_hosts_addr

def is_normal(source):
	normal_hosts = ["147.32.84.170","147.32.84.134","147.32.84.164","147.32.87.36","147.32.80.9","147.32.87.11"]

	return source in normal_hosts


