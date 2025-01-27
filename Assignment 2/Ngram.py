import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import math


# Discretize the data of the signal in the provide amount of groups
def dicretize_data(data, signal, discretize_groups):
    # Initiliaze arrays     
    label_classes = []
    step_sizes = []
    
    # Compute min and max value of signal to compute the stepsize     
    min_value = min(data[signal])
    max_value = max(data[signal])
    step_size = (max_value - min_value) / discretize_groups

    bins = []
    bins.append(min_value)
    # Create bins      
    for i in range(discretize_groups):
        label_classes.append(i)
        bins.append(bins[i] + step_size)
    
    # Use digitize method to discretize the signal
    return np.digitize(data[signal], bins), bins

# Plot the discretization of the data
def plot_discretize(data, signal, discretize_groups, bins, type_data):
    plt.title("Discretized " + signal + " signal ("+ type_data + ")  - " + str(discretize_groups) + "  groups")
    plt.xlabel("Groups")
    plt.ylabel("Frequency")
    plt.hist(data)
    plt.show()

# Computes the occurences of each n-gram for the provided amount of sliding windows
def compute_ngram_matrix(data, data_in_buckets, signal, sliding_windows, n, stepsize):
    # Initialize matrix with amounnt of rows equal to the amount of sliding windows     
    matrix = pd.DataFrame(np.zeros(len(sliding_windows)))
    # Initialize counters     
    count_sliding_window = 0
    
    # For each sliding window length compute the occurences of the n-grams     
    for sliding_window in sliding_windows:
        i = 0
        print(f"it: {count_sliding_window}")
        # Pick part of signal(depending on size of sliding window) until end of signal is reached
        while i + sliding_window <= len(data[signal]):
            # Pick part of signal with size of sliding window
            cur_sliding_window = data_in_buckets[i:i + sliding_window]
            j = 0
            # Loop through part of signal and count the n-grams until end is reached             
            while j + n <= len(cur_sliding_window):
                # Current n-gram                 
                cur_ngram = cur_sliding_window[j:j + n]
                cur_ngram_string = ''
                # Create string value of n-gram                 
                for k in cur_ngram:
                    cur_ngram_string += str(k)
                # Check if matrix already contains a column for the specific n-gram, if so, add 1 to the correct cell.
                # If not, create a column and set                  
                if cur_ngram_string not in matrix.columns:
                    matrix.loc[count_sliding_window, cur_ngram_string] = 1
                else:
                    if math.isnan(matrix.loc[count_sliding_window, cur_ngram_string]):
                        matrix.loc[count_sliding_window, cur_ngram_string] = 1
                    else:
                        matrix.loc[count_sliding_window, cur_ngram_string] += 1
                j += 1
            i += stepsize
        count_sliding_window += 1
    # To avoid NAN's, replace all NAN's with 0      
    matrix = matrix.fillna(0)
    # Delete first column(was there just for initialization of the dataframe)     
    matrix = matrix.drop(matrix.columns[0], axis=1)
   
    # Normalize all rows by their total value
    matrix = matrix.div(matrix.sum(axis=1), axis=0)
    return matrix
# Returns the L most frequent n-grams in the data
def get_most_freq_ngrams(data, threshold):
    # Count the total occurence of each N-gram 
    top_L = data.sum(axis=0)
    # Sort the occurences of each n-gram in descending order
    top_L = top_L.sort_values(ascending=False)
    # Compute the total occurences for normalization
    sum_all_values = top_L.sum()
    # Normalize
    top_L = top_L.divide(sum_all_values)
    # Pick top X n-grams
    top_L = top_L[0:threshold].index
    return top_L

# Function to plot the anomalies in the specified range
def plot_anomaly(signal_test, signal, indexes_of_signal, start, end):
    # y-values     
    y = np.array(signal_test[signal][start:end])
    list_xvalues = list(range(start,end)) 
    x = np.array(list_xvalues)

    # Anamolies should be marked red
    threshold = 15
    plt.figure()
    # Plot the line
    plt.plot(x, y, color='blue')
    plt.title("Anomaly detection for signal " + str(signal))
    plt.xlabel("Time")
    plt.ylabel("Value of signal")
    plt.scatter(x[np.in1d(x,indexes_of_signal)], y[np.in1d(x,indexes_of_signal)], color='red')