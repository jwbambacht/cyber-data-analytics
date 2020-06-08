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
    
#     discretized_counter = {}
#     for key, value in sorted_counter.items():
#         current_encoding = [int(key[0]), int(key[1]), int(key[2])]
#         # Find old encoding         
#         old_encoding = le.inverse_transform(current_encoding)
#         new_key = ','.join(old_encoding)
#         # Replace new encoding with old encoding, to understand the meaning of the n-grams
#         discretized_counter[new_key] = value
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
    
#     freq_counter = {}
#     for key, value in sorted_counter.items():
#         current_encoding = [int(key[0]), int(key[1]), int(key[2])]
#         # Find old encoding         
#         old_encoding = le.inverse_transform(current_encoding)
#         new_key = ','.join(old_encoding)
#         # Replace new encoding with old encoding, to understand the meaning of the n-grams
#         freq_counter[new_key] = value
    return sorted_counter 





# def plot_ngram_frequency(data, counter_size):
#     # Create dictionary only containing top 10 most frequent n-grams(dictionary already sorted in decreasing value order)
#     top_ten_items = data.items()
#     top_ten_items = list(top_ten_items)[:10]
#     top_ten = {}
#     for i in top_ten_items:
#         top_ten[i[0]] = i[1]


#     # Create dataframe for the plot, where x-axes is ngram profile and y-axes is the frequency count     
#     plot_df = pd.DataFrame()
#     plot_df['ngram'] = top_ten.keys()
#     plot_df['frequency'] = top_ten.values()
#     fig, ax = plt.subplots(ncols=1,figsize=(22,10))
#     sns.barplot(x="ngram", data=plot_df, y='frequency')
#     ax.set_title("10 most frequent 3-grams using " + str(counter_size) + " counters - Non Infected Host: Protocol-Duration feature", size=16)
#     plt.xticks(rotation=90)
#     plt.show()
