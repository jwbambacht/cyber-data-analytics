import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import random
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from scipy import interp


# Computes the ROC curve
def compute_roc(classifier, test_input, test_output, classifier_name, mean_tpr, mean_fpr, fprArray, tprArray):
    probs = classifier.predict_proba(test_input)
    preds = probs[:, 1]
    fpr, tpr, threshold = metrics.roc_curve(test_output, preds)
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    fprArray[classifier_name].append(fpr)
    tprArray[classifier_name].append(tpr)
    return mean_tpr

# Preprocess the data for the specific columns
def preprocess(data):
    data = data.loc[~(data["simple_journal"] == "Refused")]
    data.loc[data["simple_journal"] == "Settled", "simple_journal"] = 0
    data.loc[data["simple_journal"] == "Chargeback", "simple_journal"] = 1
    data[['issuercountrycode']] = data[['issuercountrycode']].replace(np.nan, 'NaN')
    data[['shoppercountrycode']] = data[['shoppercountrycode']].replace(np.nan, 'NaN')
    data[['cardverificationcodesupplied']] = data[['cardverificationcodesupplied']].replace(np.nan, 'NaN')
    data[['cardverificationcodesupplied']] = data[['cardverificationcodesupplied']].replace(True, 'True')
    data[['cardverificationcodesupplied']] = data[['cardverificationcodesupplied']].replace(False, 'False')
    data[['bin']] = data[['bin']].replace(np.nan, -1)
    return data

# Increases the level of privacy using the Rank Swap method
def rank_swap(data, p):
    # For each column apply the rank-swap method
    for colname in data.columns:
        # Sort the column in ascending order         
        data = data.sort_values(by=[colname])
        
        # Skip the rows simple_journal and bookingdate         
        if colname != 'simple_journal' and colname != 'bookingdate':
            
            # For each row pick a random row index within the window(i-p, i+p), to keep the chosen row values somewhat close
            for i in range(len(data[colname]) - 1):
                # Check for index out of bound errors, if so set to min/max
                begin = i - p
                if begin < 0:
                    begin = 0
                end = i + p
                if end >= len(data[colname]) - 1:
                    end = len(data[colname]) - 1
                
                # Value of current row
                swapValue1 = data[colname].iloc[i]
                randRowIndex = random.randint(begin, end)
                # Value of randomly selected row
                swapValue2 = data[colname].iloc[randRowIndex]
                # Switch values
                data[colname].iloc[i] = swapValue2
                data[colname].iloc[randRowIndex] = swapValue1
    # Save data to csv file
    data.to_csv('data/shuffledData.csv', index=False)
    return data

# Method which uses SMOTE, Cross-validation and plots the performance of the six different classification methods
def plotRocs():
    # Pick correct data set
    # data = pd.read_csv("data/shuffledData.csv")
    data = pd.read_csv("data/sampledData.csv")
    
    # This code is here just for showing how the sample was taken of the whole data set
    # data = data.sample(frac=0.18, replace=True)
    # data = data.reset_index(drop=True)
    # data = rank_swap(data, 10)
    
    # Preprocess the data 
    data = preprocess(data)
    # Array containing the names of the classifcation methods
    classification_methods = ['DecisionTree', 'SVM', 'RandomForest', 'KNN', 'NaiveBayes', 'LogisticRegression']
    
    # Initialize variables     
    fprArray = dict()
    tprArray = dict()
    colors_plot = dict()
    colors = ['b', 'pruple', 'g', 'r', 'orange', 'b']
    count = 0
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    
    # For each classification method create empty key, value pairs to save the performance    
    for class_method in classification_methods:
        fprArray[class_method] = []
        tprArray[class_method] = []
        colors_plot[class_method] = colors[count]
        count += 1

    # Drop the bookingdate column since it is not fair to use this information
    X = data.drop(['bookingdate'], axis=1)
    # The column, simple_journal, is the ouput variable
    y = data.iloc[:, 9]
    # Encode data the data otherwise it could be classified
    X = X.apply(LabelEncoder().fit_transform)
    # The amount of splits used in cross-validation      
    splits = 6
    # Create test and train sets
    cv = KFold(n_splits=splits, shuffle=True)
    iteration = 0
    mean_tpr = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # For each train and test set run the different classifiers and save the results
    for train_index, test_index in cv.split(X):
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]
        X_test = X.iloc[test_index]
        y_test = y.iloc[test_index]

        # Resampling (Only training data)
        
        # SMOTE Re-sampling
        sm = SMOTE(random_state=12)
        X_train, y_train = sm.fit_sample(X_train, y_train)

        # Majority/Minority sampling 
        # train_data_minority = X_train.loc[data['simple_journal'] == 1]
        # train_data_majority = X_train.loc[data['simple_journal'] != 1]
        #
        # train_data_downsample = resample(train_data_majority,
        #                                  replace=True,
        #                                  n_samples=10000)
        #
        # data_train_upsample = resample(train_data_minority,
        #                                replace=True,
        #                                n_samples=10000)
        # data_train = pd.concat([train_data_downsample, data_train_upsample])
        # y_train = data_train.iloc[:, 8]
        # X_train = data_train.drop(['simple_journal'], axis=1)
        # X_test = X_test.drop(['simple_journal'], axis=1)
        
        # Drop the output colum         
        X_train = X_train.drop(['simple_journal'], axis=1)
        X_test = X_test.drop(['simple_journal'], axis=1)

        # Classifiers
        # Decision Tree classifer 
        clf = DecisionTreeClassifier()
        clf = clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        compute_mean_trc = compute_roc(clf, X_test, y_test, classification_methods[0], mean_tpr[0], mean_fpr, fprArray, tprArray)
        mean_tpr[0] = compute_mean_trc

        # Support vector machine
        clf = svm.SVC(gamma='scale', probability=True)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        compute_mean_trc = compute_roc(clf, X_test, y_test, classification_methods[1], mean_tpr[1], mean_fpr, fprArray, tprArray)
        mean_tpr[1] = compute_mean_trc
        
        # Random Forest
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        compute_mean_trc = compute_roc(clf, X_test, y_test, classification_methods[2], mean_tpr[2], mean_fpr, fprArray, tprArray)
        mean_tpr[2] = compute_mean_trc

        # K Nearest Neighbours:
        clf = KNeighborsClassifier(n_neighbors=2)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        compute_mean_trc = compute_roc(clf, X_test, y_test, classification_methods[3], mean_tpr[3], mean_fpr, fprArray, tprArray)
        mean_tpr[3] = compute_mean_trc

        # Gaussian Naive Bayes
        clf = GaussianNB()
        y_pred = clf.fit(X_train, y_train).predict(X_test)
        compute_mean_trc = compute_roc(clf, X_test, y_test, classification_methods[4], mean_tpr[4], mean_fpr, fprArray, tprArray)
        mean_tpr[4] = compute_mean_trc
        
        # Logistic Regression
        clf = LogisticRegression(random_state=0, max_iter=10000).fit(X_train, y_train)
        clf.predict(X_test)
        compute_mean_trc = compute_roc(clf, X_test, y_test, classification_methods[5], mean_tpr[5], mean_fpr, fprArray, tprArray)
        mean_tpr[5] = compute_mean_trc
        iteration += 1
    
    # For each simulation plot the ROC curve      
    for i in range(len(classification_methods)):
        mean_tpr[i] /= splits
        plt.plot(mean_fpr, mean_tpr[i], label='Mean ' + classification_methods[i], lw=2)
        
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic - Only sampled')
    # plt.title('Receiver operating characteristic - RankedSwapped')
    plt.legend(loc="lower right")
    plt.savefig('result_only_sampled' + '.png')
    # plt.savefig('result_shuffled' + '.png')
    plt.show()