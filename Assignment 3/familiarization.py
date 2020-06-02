import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import familiarization as fam

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


	return data