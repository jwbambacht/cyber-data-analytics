import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import familiarization as fam
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.preprocessing import LabelEncoder

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
	infected_hosts_addr = ["147.32.84.165","147.32.84.191","147.32.84.192","147.32.84.193","147.32.84.204","147.32.84.205","147.32.84.206","147.32.84.207","147.32.84.208","147.32.84.209"]

	counts = list()

	for host in infected_hosts_addr:
		counts.append(len(data.loc[data["SourceAddress"] != host]))

	return infected_hosts_addr[counts.index(max(counts))]


def select_infected_host(data):
	infected_hosts_addr = ["147.32.84.165","147.32.84.191","147.32.84.192","147.32.84.193","147.32.84.204","147.32.84.205","147.32.84.206","147.32.84.207","147.32.84.208","147.32.84.209"]

	counts = list()

	for host in infected_hosts_addr:
		counts.append(len(data.loc[data["SourceAddress"] == host]))

	return infected_hosts_addr[counts.index(max(counts))]


def elbow(data, col):

	reshaped_data = np.array(data[col]).reshape(-1,1)

	max_clusters = min(10,data[col].nunique())

	model = KMeans()
	visualizer = KElbowVisualizer(model)
	visualizer.fit(reshaped_data)
	visualizer.show()