import pandas as pd
import numpy as np
import random

# Create the minhash matrix where rows represent ip-pairs and columns the unique n-grams
# Cell value represent if the ngram is exists in the discretized stream data of the ip-pair
def create_minhash_matrix(data):
    # Initialize dataframe with dummy data(later deleted)     
    dummyRow = ['dummyRow']
    dummyCol = ["dummyCol"]
    df_minhash = pd.DataFrame(columns=dummyCol, index=dummyRow)

    # For each unique ip-pair compute ..
    for i in data['SourceAddressEncoded'].unique():
        for j in data['DestinationAddressEncoded'].unique():
            # Avoid ip-pairs with itself         
            if i != j:
                # Generate row index
                key = str(i) + "," + str(j)

                # Find all discretized values for a specifc ip-pair
                ippair = data[(data['SourceAddressEncoded'] == i) & (data["DestinationAddressEncoded"] == j)]

                # Generate ngram input for the current ip-pair
                ngramstring = ''
                for k in ippair['single_decretization']:
                    ngramstring += str(k)
                # Set correct cell to 1 of ip-pair contains n-gram
                for j in range(0, len(ngramstring) - 2):
                    cur_ngram = ngramstring[j:j+3]
                    if cur_ngram not in df_minhash.columns:
                        df_minhash.loc[key, cur_ngram] = 1
                    else:
                        if key not in df_minhash.index:
                            df_minhash.loc[key, cur_ngram] = 1

    # Drop dummy columns and replace NAN's                             
    df_minhash= df_minhash.drop(['dummyRow'])
    df_minhash= df_minhash.drop(['dummyCol'], axis=1)
    df_minhash = df_minhash.fillna(0)
    return df_minhash

def create_hash_signature_matrix(df_minhash, amount_of_hash_functions):
    # Initialize data     
    columns_index = range(0, len(df_minhash.index))
    hash_functions = range(0, amount_of_hash_functions)
    
    # Create dataframe for signature matrix where row represent the hash functions and columns the ip-pairs profiles     
    df_sig = pd.DataFrame(columns=columns_index, index=hash_functions, dtype='int32')
    
    # Set all signature values to infinity
    df_sig = df_sig.fillna(np.inf)
   
    size_of_buckets = len(df_minhash.columns)

    # Reset column and row names
    df_minhash.columns = range(0, len(df_minhash.columns))
    df_minhash.index = range(0, len(df_minhash.index))
    random_numbers_for_hash = []

    # Create X random hash functions save them as (random number 1, random number 2)
    for i in hash_functions:
        r1 = random.randint(1,10)
        r2 = random.randint(1,10)
        random_numbers_for_hash.append([r1,r2])

    # For each column in the minhash table compute hash values and save lowest if cell value of row,column is 1     
    for col in df_minhash.columns:

        # Calculate value for random hashes
        hashes = []
        for i in hash_functions:
            calc_hash = (random_numbers_for_hash[i][0] * col + random_numbers_for_hash[i][1]) % size_of_buckets
            hashes.append(calc_hash)

        # Set the value of the hash value if below current saved value and cell value of minhash is 1     
        for row in range(0, len(df_sig.columns)):
            # Check if ip-pair contains n-gram         
            if df_minhash.loc[row][col] == 1:
                # Save value for each hash if it mets condition             
                for h in range(0, len(hashes)):
                    # Save lowest values for each hash function
                    if df_sig.loc[h][col] > hashes[h]:
                        df_sig.loc[h][col] = int(hashes[h])
    return df_sig