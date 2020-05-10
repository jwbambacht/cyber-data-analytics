import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, plot_roc_curve, confusion_matrix
import random
import classification as clf

sns.set_style("darkgrid")

def load_data(file_name):
	data = pd.read_csv(file_name)
	data.head()

	return data

def get_fraud(data):
	return data.loc[data['simple_journal'] == 1]

def get_non_fraud(data):
	return data.loc[data['simple_journal'] == 0]

def describe_data(data):
	total_size = len(data)
	fraud_size = len(clf.get_fraud(data))
	non_fraud_size = len(clf.get_non_fraud(data))

	fraud_ratio = round(fraud_size/len(data)*100,4)
	non_fraud_ratio = round(non_fraud_size/len(data)*100,4)
	
	print(f"Total: {total_size}")
	print(f"Non-Fraud: {non_fraud_size} = {non_fraud_ratio}%")
	print(f"Fraud: {fraud_size} = {fraud_ratio}%\n")

def pre_process(data):
	# The class label needs to be determined. 
	# We can neglect the refused cases since we don't know if it was fraud or not.
	# We assign class 1 for Chargeback and class 0 for Settled
	pdata = data.loc[~(data["simple_journal"] == "Refused")]
	pdata.loc[pdata["simple_journal"] == "Settled", "simple_journal"] = 0
	pdata.loc[pdata["simple_journal"] == "Chargeback", "simple_journal"] = 1

	# For every occurrence we generalize the local currency amount by converting to euros
	currency_convert = {"GBP": 0.88, "AUD": 1.67, "SEK": 10.62, "MXN": 26.03, "NZD": 1.78}
	data["amount"] = data.apply(lambda x: int(x["amount"]/currency_convert[x["currencycode"]]),axis=1)

	# Manually encode this feature into three values
	pdata.loc[pdata["shopperinteraction"] == "Ecommerce", "shopperinteraction"] = 0
	pdata.loc[pdata["shopperinteraction"] == "ContAuth", "shopperinteraction"] = 1
	pdata.loc[pdata["shopperinteraction"] == "POS", "shopperinteraction"] = 2

	# Set category 3-6 to 3 since it doesn't matter what it is among them
	pdata.loc[pdata["cvcresponsecode"] > 3, "cvcresponsecode"] = 3

	# Automatically encode categorical columns to be applicable in the classifiers
	le = LabelEncoder()
	pdata['issuercountrycode'] = le.fit_transform(pdata['issuercountrycode'].astype(str))
	pdata['txvariantcode'] = le.fit_transform(pdata['txvariantcode'])
	pdata['currencycode'] = le.fit_transform(pdata['currencycode'].astype(str))
	pdata['shoppercountrycode'] = le.fit_transform(pdata['shoppercountrycode'].astype(str))
	pdata['accountcode'] = le.fit_transform(pdata['accountcode'])
	pdata['mail_id'] = le.fit_transform(pdata['mail_id'])
	pdata['ip_id'] = le.fit_transform(pdata['ip_id'])
	pdata['card_id'] = le.fit_transform(pdata['card_id'])

	# Remove columns since they do not add value to the classification
	pdata.drop(columns=['txid','bookingdate','bin','creationdate','cardverificationcodesupplied'],axis=1,inplace=True)

	# Change any value that can't be used by the classifiers
	pdata = clean_dataset(pdata)

	return pdata

def clean_dataset(data):
	assert isinstance(data, pd.DataFrame), "df needs to be a pd.DataFrame"
	data.dropna(inplace=True)
	indices_to_keep = ~data.isin([np.nan, np.inf, -np.inf]).any(1)
	return data[indices_to_keep].astype(np.float64)

def get_fraud(data):
	return data.loc[data['simple_journal'] == 1]

def get_non_fraud(data):
	return data.loc[data['simple_journal'] == 0]

def get_X_y(data):
	y = np.array(data['simple_journal'])
	X = np.array(data.drop(columns='simple_journal'))

	return X,y

def nearest_neighbor(data):
	nn = NearestNeighbors(n_neighbors=5, metric='euclidean', algorithm='kd_tree').fit(data)
	dist, index = nn.kneighbors(data)
	return index

def smote_item(data):
	indices = nearest_neighbor(data)

	result = []

	for m in range(len(indices)):
		t = data[indices[m]]
		new_t = pd.DataFrame(t)
		result.append([])

		for j in range(len(new_t.columns)):
			result[m].append(random.choice(new_t[j]))

	return result

def smote_dataset(X_UNSMOTEd, y_UNSMOTEd,N):

	X_UNSMOTEd = np.array(X_UNSMOTEd)
	y_UNSMOTEd = np.array(y_UNSMOTEd)

	X_SMOTEd = X_UNSMOTEd
	y_SMOTEd = y_UNSMOTEd

	for i in range(N):
		unique, counts = np.unique(y_UNSMOTEd,return_counts=True)
		minority_shape = dict(zip(unique, counts))[1]

		x = np.ones((minority_shape, X_UNSMOTEd.shape[1]))
		x = [X_UNSMOTEd[i] for i, v in enumerate(y_UNSMOTEd) if v == 1.0]
		x = np.array(x)

		X_sampled = smote_item(x)
		X_SMOTEd = np.concatenate((X_SMOTEd, X_sampled), axis=0)

		y_sampled = np.ones(minority_shape)
		y_SMOTEd = np.concatenate((y_SMOTEd, y_sampled), axis=0)

	return X_SMOTEd, y_SMOTEd

def set_threshold(y_prob, threshold):
	y_pred = y_prob[:,1]
	for i in range(len(y_prob)):
		if(y_prob[i,1] >= threshold):
			y_pred[i] = 1
		else:
			y_pred[i] = 0

	return y_pred

def classify_knn(data, N):
	X, y = clf.get_X_y(data)
	kf = StratifiedKFold(n_splits=10, shuffle=True,random_state=13)
	# kf = KFold(n_splits=10,random_state=33)

	classifier = KNeighborsClassifier(n_neighbors=3)

	tp_total = 0
	tn_total = 0
	fp_total = 0
	fn_total = 0

	for train_index, test_index in kf.split(X, y):
	# for train_index, test_index in kf.split(X):
		X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]

		X_train, y_train = clf.smote_dataset(X_train, y_train, N)

		classifier.fit(X_train,y_train)
		y_prob = classifier.predict_proba(X_test)
		y_pred = clf.set_threshold(y_prob,0.51)

		tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()
		tp_total += tp
		tn_total += tn
		fp_total += fp
		fn_total += fn

	print(f"TruePositives: {tp_total}")
	print(f"TrueNegatives: {tn_total}")
	print(f"FalsePositives: {fp_total}")
	print(f"FalseNegatives: {fn_total}")

def classify_random_forest(classifierName, data, N):
	X, y = clf.get_X_y(data)
	X, y = clf.smote_dataset(X, y, N)
	kf = KFold(n_splits=10,shuffle=True,random_state=13)

	classifier = RandomForestClassifier(random_state=0, n_estimators=100)

	tp_total = 0
	tn_total = 0
	fp_total = 0
	fn_total = 0

	for train_index, test_index in kf.split(X):
		X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]

		classifier.fit(X_train,y_train)
		y_pred = classifier.predict(X_test)

		tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()
		tp_total += tp
		tn_total += tn
		fp_total += fp
		fn_total += fn

	print(f"TruePositives: {tp_total}")
	print(f"TrueNegatives: {tn_total}")
	print(f"FalsePositives: {fp_total}")
	print(f"FalseNegatives: {fn_total}")

def get_performance(clf, predicted, y_test,name):

	print(f"{name}:")
	print(f"Accuracy: {accuracy_score(y_test, predicted)}")
	print(f"Precision: {precision_score(y_test, predicted)}")
	print(f"Recall: {recall_score(y_test, predicted)}")
	print(f"F1: {f1_score(y_test, predicted)}\n")




	

	

	






