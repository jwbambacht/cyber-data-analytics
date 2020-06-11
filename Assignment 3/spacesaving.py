import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from itertools import islice

def SpaceSaving(data, k): 
    n_gram_length = 3
    k = k + 1
    
    # Create one input stream containing all the discretized data of the combined features
    ngram_input = ''
    for i in data:
        ngram_input += str(i)

    # Initialize values
    counter_list = {}
    T = []
    
    # For each n-gram of size 3 until end of input is reached count the elements based on the SpaceSaving algorithm
    for i in range(0, len(ngram_input) - n_gram_length+1):
        # Pick current ngram
        cur_ngram = ngram_input[i: i+n_gram_length]

        # Space save algorithm     
        if cur_ngram in T:
            # Check if the ngram is contained in the list T, if so, increment counter
            counter_list[cur_ngram] += 1
        elif len(T) < k - 1:
            # Check if length of T is smaller than k-1, if so, add the ngram to list
            # And set the counter for that ngram to 1
            T.append(cur_ngram)
            counter_list[cur_ngram] = 1
        else:
            # Get the key containing the min value in the counter dictionary
            min_value_key = min(counter_list, key=counter_list.get)
            # Get the value of the minimum position
            min_value = counter_list[min_value_key]

            # Check if the current ngram is already contained in the counter dictionary to avoid undefined errors
            counter_list[cur_ngram] = min_value + 1

            # Update T and keep the counter dictionary fixed size(k)
            T.remove(min_value_key)
            counter_list.pop(min_value_key)
            # Replace by new value             
            T.append(cur_ngram)

    # Sort the frequency in descending order     
    sorted_counter = {k: v for k, v in sorted(counter_list.items(), key=lambda item: item[1], reverse=True)}
    return sorted_counter 

def ActualFrequency(data): 
    n_gram_length = 3

    # Create one input stream containing all the discretized data of the combined features
    ngram_input = ''
    for i in data:
        ngram_input += str(i)

    # Initialize values
    freq_dict = {}
    
    # For each n-gram of size 3 until end of input is reached count the elements based on the SpaceSaving algorithm
    for i in range(0, len(ngram_input) - n_gram_length+1):
        # Pick current ngram
        cur_ngram = ngram_input[i: i+n_gram_length]
        if cur_ngram in freq_dict:
            freq_dict[cur_ngram] += 1
        else:
            freq_dict[cur_ngram] = 1
         
    # Sort the frequency in descending order     
    sorted_counter = {k: v for k, v in sorted(freq_dict.items(), key=lambda item: item[1], reverse=True)}
    return sorted_counter
