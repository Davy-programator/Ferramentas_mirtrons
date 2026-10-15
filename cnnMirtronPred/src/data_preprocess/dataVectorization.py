""" zero padding the RNA sequences into the maximum length,
    vectorize the RNA sequences with one-hot coding
"""
import numpy as np
import dataRead

def vectorize_data(dataset, max_seq_len = 0):
    # get the maxmium length of the seqence
    # max_seq_len = 0
    for item in dataset:
        if len(item[1])>max_seq_len:
            max_seq_len = len(item[1])
    #print(max_seq_len)

    # padding with "N" to max_seq_len
    for item in dataset:
        item[1] += "N" *(max_seq_len-len(item[1]))
 
    # tranformation of data set:one_hot encoding
    x_cast = {"A":[[1],[0],[0],[0]],"U":[[0],[1],[0],[0]],\
              "T":[[0],[1],[0],[0]],"G":[[0],[0],[1],[0]],\
              "C":[[0],[0],[0],[1]],"N":[[0],[0],[0],[0]]}
    y_cast = {"TRUE": [1,0],"FALSE":[0,1]} #TRUE:Mirtrons  FALSE:canonical microRN
    
    # define a list to store the vectorized data
    vectorized_dataset = []
    
    for item in dataset:
         data = []
         for char in item[1]:
             data.append(x_cast[char])
         vectorized_dataset.append([item[0],data,y_cast[item[2]]])
    
    return vectorized_dataset

if __name__ == "__main__":
    FILE_PATH = "../../data/miRBase_set.csv"
    FILE_PATH_PUTATIVE = "../../data/putative_mirtrons_set.csv"
    all_data_array = dataRead.read_data(FILE_PATH,FILE_PATH_PUTATIVE)
    vectorized_dataset = vectorize_data(all_data_array)
    print(vectorized_dataset)
