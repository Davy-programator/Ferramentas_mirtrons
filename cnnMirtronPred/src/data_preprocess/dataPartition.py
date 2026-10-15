"""Partition the dataset into X_train, y_train, X_test, y_test
"""
import numpy as np
import dataVectorization
import dataRead

def data_partition(data_vectors):
    X_train, y_train, X_test, y_test = [],[],[],[]
    i,j = 0,0
    for item in data_vectors:
        if item[2]==[1,0]:
            i = i + 1
            if i <= 201:
                X_test.append(item[1])
                y_test.append(item[2])
            else:
                    X_train.append(item[1])
                    y_train.append(item[2])
        else:
            j = j + 1
            if  j<= 200:
                X_test.append(item[1])
                y_test.append(item[2])
            else:
                X_train.append(item[1])
                y_train.append(item[2])

    return X_train, y_train, X_test, y_test

def data_partition_2(data_vectors):
    X_train, y_train = [], []
    for item in data_vectors:
        X_train.append(item[1])
        y_train.append(item[2])

    return X_train, y_train, X_train, y_train

if __name__ == "__main__":
    FILE_PATH = "../../data/miRBase_set.csv"
    FILE_PATH_PUTATIVE = "../../data/putative_mirtrons_set.csv"
    all_data_array = dataRead.read_data(FILE_PATH,FILE_PATH_PUTATIVE)
    vectorized_dataset = dataVectorization.vectorize_data(all_data_array)
    X_train, y_train, X_test, y_test = data_partition(vectorized_dataset)   
    print(X_train)
