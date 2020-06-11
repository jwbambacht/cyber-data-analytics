import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import lsh_hyperplane as lsh
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
def pre_process(data):
	# Rename label column for usability, and remove background flows
	data.rename(columns={"Dur":"Duration","Proto":"Protocol","SrcAddr":"SourceAddress","Sport":"SourcePort","Dir":"Direction","DstAddr":"DestinationAddress","Dport":"DestinationPort","TotPkts":"TotalPackets","TotBytes":"TotalBytes","SrcBytes":"SourceBytes","Label(Normal:CC:Background)": "Label"}, inplace=True)
	data = data[~data.Label.str.contains("Background")]

	# Remove useless columns
	data = data.drop(columns=["sTos","dTos","State"],axis=1)

	# Create column that indicates the infected and non-infected hosts
	infected_hosts_addr = ["147.32.84.165","147.32.84.191","147.32.84.192","147.32.84.193","147.32.84.204","147.32.84.205","147.32.84.206","147.32.84.207","147.32.84.208","147.32.84.209"]
	data["Infected"] = 0
	data.loc[data["SourceAddress"].isin(infected_hosts_addr),"Infected"] = 1

	data['Protocol'] = data['Protocol'].str.upper() 

	return data

def encode_feature(data):
	le = LabelEncoder()
	data = le.fit_transform(data)

	return data

def select_non_infected_host(data):
	data = data.loc[data["Infected"] == 0]

	non_infected_hosts = data.SourceAddress.unique()

	counts = list()

	for host in non_infected_hosts:
		counts.append(len(data.loc[data["SourceAddress"] == host]))

	return non_infected_hosts[counts.index(max(counts))]


def select_infected_host(data):
	infected_hosts_addr = ["147.32.84.165","147.32.84.191","147.32.84.192","147.32.84.193","147.32.84.204","147.32.84.205","147.32.84.206","147.32.84.207","147.32.84.208","147.32.84.209"]

	counts = list()

	for host in infected_hosts_addr:
		counts.append(len(data.loc[data["SourceAddress"] == host]))

	return infected_hosts_addr[counts.index(max(counts))]


def elbow(data, col):

	reshaped_data = np.array(data[col]).reshape(-1,1)

	if col == "Protocol":
		k = (2,6)
	else:
		k = (2,11)

	model = KMeans()
	visualizer = KElbowVisualizer(model, k=k)
	visualizer.fit(reshaped_data)
	visualizer.show()

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

def concatenate_columns(column_one, column_two, column_three):
	# Concatenates the values of two columns
	return str(int(column_one))+""+str(int(column_two))+""+str(int(column_three))


def hash_function(value, i, width):
	random.seed(3)
	seed_one = random.randint(0,1000)
	seed_two = random.randint(0,1000)

	hash_one = hash(value)*seed_one
	hash_two = hash(value)*seed_two

	if i == 0:
		return hash_one % width
	elif i == 1:
		return hash_two % width
	else:
		return (hash_one + i*hash_two) % width

def generate_random_hyperplane(size, dimension):
	return np.random.randn(size, dimension)

def dot_product(v, r):
    res = 0
    for i in range(len(v)):
        res += float(v[i])*r[i]
    
    return res

def sign_function(value):

    if value > 0:
        return 1
    
    return 0

def is_infected(ip):
	infected_hosts_addr = ["147.32.84.165","147.32.84.191","147.32.84.192","147.32.84.193","147.32.84.204","147.32.84.205","147.32.84.206","147.32.84.207","147.32.84.208","147.32.84.209"]

	if ip in infected_hosts_addr:
		return True

	return False

def is_normal(ip):
	normal_hosts_addr = ["147.32.84.170","147.32.84.134","147.32.84.164","147.32.87.36","147.32.80.9","147.32.87.11"]

	if ip in normal_hosts_addr:
		return True

	return False

def euclidean_distance(arr1, arr2):
	sum = 0

	for i in range(len(arr1)):
		sum += (arr1[i]-arr2[i])**2

	return math.sqrt(sum)




